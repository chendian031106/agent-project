"""
BaseAgent 基类

所有业务智能体的基类，提供：
- LLM 调用封装（DeepSeek V4 / Qwen 双模型）
- LangChain ConversationBufferMemory（短期记忆）
- Redis 持久化长期记忆
- LangChain Tool 调用支持
- 流式输出支持
- 统一的 Pydantic 输入/输出格式
- 内置日志和错误处理
"""

import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence

from langchain_classic.memory import ConversationBufferMemory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from utils.config import settings, get_chat_model, get_lightweight_model
from utils.logger import logger
from utils.redis_client import RedisClient

# ============ Pydantic 数据模型 ============


class AgentInput(BaseModel):
    """统一的智能体输入格式"""

    task: str = Field(description="任务描述")
    context: Optional[Dict[str, Any]] = Field(default=None, description="额外上下文信息")
    session_id: Optional[str] = Field(default=None, description="会话ID，用于记忆持久化")
    max_tokens: int = Field(default=2048, description="最大输出 token 数")
    temperature: float = Field(default=0.7, description="生成温度")


class AgentOutput(BaseModel):
    """统一的智能体输出格式"""

    success: bool = Field(default=True, description="是否成功")
    output: Optional[str] = Field(default=None, description="最终输出文本")
    streaming_output: Optional[List[str]] = Field(default=None, description="流式输出的分片")
    error: Optional[str] = Field(default=None, description="错误信息")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    token_usage: Optional[Dict[str, int]] = Field(default=None, description="Token 用量")
    latency_ms: int = Field(default=0, description="耗时（毫秒）")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="扩展元数据")


class MemoryEntry(BaseModel):
    """单条记忆记录"""

    role: str = Field(description="角色: user / assistant / system")
    content: str = Field(description="内容")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="时间戳")


class LLMConfig(BaseModel):
    """LLM 配置模型"""

    provider: str = Field(default="deepseek", description="提供商: deepseek / qwen")
    model: str = Field(default="deepseek-chat", description="模型名称")
    api_key: str = Field(default="", description="API Key")
    api_base: str = Field(default="https://api.deepseek.com/v1", description="API 地址")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048)
    top_p: float = Field(default=0.9)


# ============ LLM 工厂 ============

_DEFAULT_DEEPSEEK_CONFIG = LLMConfig(
    provider="deepseek",
    model="deepseek-chat",
    api_key=settings.DEEPSEEK_API_KEY,
    api_base=settings.DEEPSEEK_API_BASE or "https://api.deepseek.com/v1",
)

_DEFAULT_QWEN_CONFIG = LLMConfig(
    provider="qwen",
    model="qwen-plus",
    api_key=settings.DASHSCOPE_API_KEY,
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def _create_llm(
    config: Optional[LLMConfig] = None,
    lightweight: bool = False,
) -> ChatOpenAI:
    """创建 LLM 实例

    委托给 utils.config 的 get_chat_model / get_lightweight_model，
    优先使用 DeepSeek V4，若无 API Key 则回退到阿里云百炼 Qwen。
    """
    if lightweight and settings.DASHSCOPE_API_KEY:
        return get_lightweight_model()

    if config is not None:
        logger.info(
            f"初始化 LLM: provider={config.provider} | "
            f"model={config.model} | "
            f"base_url={config.api_base}"
        )
        return ChatOpenAI(
            model=config.model,
            api_key=config.api_key,
            base_url=config.api_base,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            streaming=False,
        )

    return get_chat_model()


# ============ BaseAgent ============


class BaseAgent(ABC):
    """智能体基类

    所有业务智能体必须继承此类，实现 `agent_name` 和 `system_prompt` 属性。

    使用示例:
        class MyAgent(BaseAgent):
            agent_name = "my_agent"
            system_prompt = "你是...专家..."
    """

    # ---- 子类必须重写的属性 ----
    agent_name: str = "base_agent"
    system_prompt: str = ""

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        tools: Optional[Sequence[BaseTool]] = None,
        memory_k: int = 10,
        use_lightweight: bool = False,
    ) -> None:
        """初始化智能体

        Args:
            llm_config: LLM 配置，默认使用 get_chat_model()
            tools: 可绑定的 LangChain 工具列表
            memory_k: 短期记忆保留轮数（对话轮次）
            use_lightweight: 是否使用轻量模型（qwen-turbo），默认 False
        """
        # LLM
        self.llm: ChatOpenAI = _create_llm(llm_config, lightweight=use_lightweight)
        self.llm_config = llm_config or (
            _DEFAULT_DEEPSEEK_CONFIG if settings.DEEPSEEK_API_KEY else _DEFAULT_QWEN_CONFIG
        )

        # 工具
        self.tools: List[BaseTool] = list(tools or [])

        # 短期记忆（按 session_id 分片）
        self._short_term_memory: Dict[str, ConversationBufferMemory] = {}

        # 短期记忆保留轮数
        self._memory_k = memory_k

        # 长期记忆（Redis）
        self._redis: Optional[RedisClient] = None
        try:
            self._redis = RedisClient()
        except Exception as e:
            logger.warning(f"[{self.agent_name}] Redis 不可用，长期记忆功能禁用: {e}")

        self._memory_key = "history"

        logger.info(
            f"[{self.agent_name}] 初始化完成 | "
            f"model={self.llm_config.model} | "
            f"tools={[t.name for t in self.tools]} | "
            f"memory_k={memory_k} | "
            f"redis={'可用' if self._redis else '不可用'}"
        )

    # ---------- 记忆管理 ----------

    @property
    def _memory_prefix(self) -> str:
        """Redis key 前缀"""
        return f"agent_memory:{self.agent_name}"

    def _get_memory(self, session_id: str) -> ConversationBufferMemory:
        """获取指定会话的短期记忆"""
        if session_id not in self._short_term_memory:
            self._short_term_memory[session_id] = ConversationBufferMemory(
                memory_key=self._memory_key,
                return_messages=True,
                k=self._memory_k,
            )
            # 尝试从 Redis 恢复长期记忆
            long_term = self._load_long_term(session_id)
            if long_term:
                for entry in long_term:
                    if entry.role == "user":
                        self._short_term_memory[session_id].chat_memory.add_user_message(
                            entry.content
                        )
                    elif entry.role == "assistant":
                        self._short_term_memory[session_id].chat_memory.add_ai_message(
                            entry.content
                        )
                logger.debug(
                    f"[{self.agent_name}] 已从 Redis 恢复 {len(long_term)} 条记忆"
                    f" (session={session_id})"
                )
        return self._short_term_memory[session_id]

    def _load_long_term(self, session_id: str) -> List[MemoryEntry]:
        """从 Redis 加载长期记忆"""
        if not self._redis:
            return []
        try:
            raw = self._redis.get(f"{self._memory_prefix}:{session_id}")
            if raw:
                data = json.loads(raw)
                return [MemoryEntry(**entry) for entry in data]
        except Exception as e:
            logger.warning(f"[{self.agent_name}] 加载长期记忆失败: {e}")
        return []

    def _save_long_term(self, session_id: str, entries: List[MemoryEntry]) -> None:
        """保存记忆到 Redis（长期持久化）"""
        if not self._redis:
            return
        try:
            # 合并已有的和新记忆
            existing = self._load_long_term(session_id)
            all_entries = (existing + entries)[-100:]  # 最多保留100条
            data = [e.model_dump() for e in all_entries]
            self._redis.set(
                f"{self._memory_prefix}:{session_id}",
                json.dumps(data, ensure_ascii=False),
                ex=86400 * 7,  # 7天过期
            )
        except Exception as e:
            logger.warning(f"[{self.agent_name}] 保存长期记忆失败: {e}")

    def _append_to_memory(
        self, session_id: str, user_msg: str, assistant_msg: str
    ) -> None:
        """将一轮对话追加到记忆"""
        mem = self._get_memory(session_id)
        mem.chat_memory.add_user_message(user_msg)
        mem.chat_memory.add_ai_message(assistant_msg)

        # 同步到 Redis
        self._save_long_term(session_id, [
            MemoryEntry(role="user", content=user_msg),
            MemoryEntry(role="assistant", content=assistant_msg),
        ])

    def clear_memory(self, session_id: str) -> None:
        """清除指定会话的记忆"""
        self._short_term_memory.pop(session_id, None)
        if self._redis:
            self._redis.delete(f"{self._memory_prefix}:{session_id}")
        logger.info(f"[{self.agent_name}] 已清除会话记忆: {session_id}")

    # ---------- 消息构建 ----------

    def _build_messages(self, task: str, context: Optional[Dict] = None) -> tuple:
        """构建消息列表

        Returns:
            (system_message, history_messages, user_message)
        """
        system = SystemMessage(content=self.system_prompt)
        user_msg = HumanMessage(content=task)
        return system, user_msg

    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        """获取工具定义（OpenAI tool call 格式）"""
        if not self.tools:
            return []
        definitions = []
        for tool in self.tools:
            try:
                # LangChain BaseTool 转 OpenAI tool schema
                schema = tool.get_input_schema().model_json_schema()
                definitions.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": schema,
                    },
                })
            except Exception as e:
                logger.warning(f"[{self.agent_name}] 工具 {tool.name} schema 解析失败: {e}")
        return definitions

    # ---------- 核心运行方法 ----------

    def run(self, input_data: AgentInput) -> AgentOutput:
        """同步运行智能体

        支持：
        - 多轮对话记忆
        - 工具自动调用
        - 错误捕获和结构化输出
        """
        start_time = time.time()
        session_id = input_data.session_id or "default"

        logger.info(
            f"[{self.agent_name}] 开始运行 | session={session_id} | "
            f"task='{input_data.task[:80]}...'"
        )

        try:
            # 1. 获取历史记忆
            if session_id != "default":
                mem = self._get_memory(session_id)
                history = mem.load_memory_variables({}).get(self._memory_key, [])
            else:
                history = []

            # 2. 构建消息
            system, user_msg = self._build_messages(input_data.task, input_data.context)
            messages = [system]
            if history:
                messages.extend(history)
            messages.append(user_msg)

            # 3. 异步执行（含工具调用循环）
            response = self._execute_with_tools(messages, input_data)

            output_text = response.get("content", "")
            token_usage = response.get("token_usage", {})

            # 4. 保存记忆
            if session_id != "default":
                self._append_to_memory(session_id, input_data.task, output_text)

            elapsed = int((time.time() - start_time) * 1000)
            logger.info(
                f"[{self.agent_name}] 运行完成 | session={session_id} | "
                f"耗时={elapsed}ms | tokens={token_usage.get('total_tokens', 'N/A')}"
            )

            return AgentOutput(
                success=True,
                output=output_text,
                session_id=session_id,
                token_usage=token_usage if token_usage else None,
                latency_ms=elapsed,
            )

        except Exception as e:
            elapsed = int((time.time() - start_time) * 1000)
            logger.error(
                f"[{self.agent_name}] 运行失败 | session={session_id} | "
                f"耗时={elapsed}ms | error={type(e).__name__}: {e}"
            )

            return AgentOutput(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                session_id=session_id,
                latency_ms=elapsed,
            )

    async def run_stream(self, input_data: AgentInput) -> AsyncGenerator[AgentOutput, None]:
        """流式运行智能体

        以异步生成器方式返回流式输出分片。
        支持 SSE 等实时推送场景。
        """
        start_time = time.time()
        session_id = input_data.session_id or "default"

        logger.info(
            f"[{self.agent_name}] 开始流式运行 | session={session_id} | "
            f"task='{input_data.task[:80]}...'"
        )

        chunks: List[str] = []
        full_output = ""
        error = None

        try:
            system, user_msg = self._build_messages(input_data.task, input_data.context)
            messages = [system, user_msg]

            # 创建流式 LLM
            streaming_llm = _create_llm(self.llm_config)
            streaming_llm.streaming = True
            streaming_llm.temperature = input_data.temperature
            streaming_llm.max_tokens = input_data.max_tokens

            async for chunk in streaming_llm.astream(messages):
                if chunk.content:
                    chunks.append(chunk.content)
                    full_output += chunk.content
                    yield AgentOutput(
                        success=True,
                        output=chunk.content,
                        streaming_output=chunks[:],
                        session_id=session_id,
                        latency_ms=int((time.time() - start_time) * 1000),
                    )

            # 流式结束，保存记忆
            if session_id != "default" and full_output:
                self._append_to_memory(session_id, input_data.task, full_output)

        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
            logger.error(
                f"[{self.agent_name}] 流式运行失败 | session={session_id} | error={error}"
            )

        # 最终状态
        elapsed = int((time.time() - start_time) * 1000)
        yield AgentOutput(
            success=error is None,
            output=full_output if error is None else None,
            streaming_output=chunks,
            error=error,
            session_id=session_id,
            latency_ms=elapsed,
        )

    def _execute_with_tools(
        self,
        messages: List,
        input_data: AgentInput,
        max_tool_rounds: int = 5,
    ) -> Dict[str, Any]:
        """执行含工具调用的 LLM 请求（同步）

        自动处理 LLM 请求工具调用的循环：
        1. LLM 返回工具调用请求
        2. 执行工具
        3. 将结果返回给 LLM
        4. 重复直到 LLM 返回最终文本

        Args:
            messages: 消息列表
            input_data: 输入参数
            max_tool_rounds: 最大工具调用轮数

        Returns:
            {"content": str, "token_usage": dict}
        """
        tool_defs = self._get_tool_definitions()
        tool_map = {t.name: t for t in self.tools}

        current_messages = list(messages)
        total_tokens = 0

        for _round in range(max_tool_rounds):
            # 调用 LLM
            kwargs = {
                "temperature": input_data.temperature,
                "max_tokens": input_data.max_tokens,
            }
            if tool_defs:
                kwargs["tools"] = tool_defs
                kwargs["tool_choice"] = "auto"

            response = self.llm.invoke(current_messages, **kwargs)
            response_msg = response

            # 统计 token
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                total_tokens += response.usage_metadata.get("total_tokens", 0)

            # 检查是否有工具调用
            if hasattr(response_msg, "tool_calls") and response_msg.tool_calls:
                current_messages.append(response_msg)

                for tc in response_msg.tool_calls:
                    tool_name = tc.get("name") or tc.get("function", {}).get("name", "")
                    tool_args_str = tc.get("args") or tc.get("function", {}).get("arguments", "{}")

                    if isinstance(tool_args_str, str):
                        try:
                            tool_args = json.loads(tool_args_str)
                        except json.JSONDecodeError:
                            tool_args = {}
                    else:
                        tool_args = tool_args_str

                    if tool_name in tool_map:
                        tool = tool_map[tool_name]
                        logger.debug(
                            f"[{self.agent_name}] 调用工具: {tool_name} | args={json.dumps(tool_args, ensure_ascii=False)[:200]}"
                        )

                        try:
                            tool_result = tool.run(tool_args)
                            if not isinstance(tool_result, str):
                                tool_result = json.dumps(tool_result, ensure_ascii=False)
                        except Exception as e:
                            tool_result = json.dumps({"error": str(e)}, ensure_ascii=False)

                        current_messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": tool_result,
                        })
                    else:
                        logger.warning(
                            f"[{self.agent_name}] 工具不存在: {tool_name} "
                            f"(可用: {list(tool_map.keys())})"
                        )
                        current_messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": json.dumps({"error": f"未知工具: {tool_name}"}),
                        })

                continue  # 继续下一轮

            # 无工具调用，返回最终文本
            return {
                "content": response_msg.content or "",
                "token_usage": {
                    "total_tokens": total_tokens,
                    "prompt_tokens": getattr(response, "usage_metadata", {}).get("prompt_tokens", 0) if hasattr(response, "usage_metadata") else 0,
                    "completion_tokens": getattr(response, "usage_metadata", {}).get("completion_tokens", 0) if hasattr(response, "usage_metadata") else 0,
                },
            }

        # 超过最大工具调用轮数
        logger.warning(f"[{self.agent_name}] 工具调用超过最大轮数 {max_tool_rounds}")
        return {
            "content": "工具调用次数过多，已自动终止。",
            "token_usage": {"total_tokens": total_tokens},
        }

    # ---------- 抽象方法 ----------

    @property
    @abstractmethod
    def agent_name(self) -> str: ...

    @property
    @abstractmethod
    def system_prompt(self) -> str: ...

    @abstractmethod
    def execute(self, task: str, **kwargs: Any) -> AgentOutput:
        """执行任务的便捷入口

        子类可在此方法中封装具体的业务逻辑。
        """
        ...


# ============ 便捷工具函数 ============


def create_agent_input(
    task: str,
    context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> AgentInput:
    """快捷创建 AgentInput"""
    return AgentInput(
        task=task,
        context=context,
        session_id=session_id,
        max_tokens=max_tokens,
        temperature=temperature,
    )