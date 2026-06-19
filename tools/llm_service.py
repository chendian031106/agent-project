"""
LLM 服务封装工具

统一封装与 DeepSeek / DashScope 等 LLM 的交互。
支持内容分析、知识问答等。
"""

import json
from typing import Any, Dict, List, Optional
from utils.logger import logger
from utils.config import settings


class LLMService:
    """大语言模型服务封装"""

    def __init__(self, model: str = "deepseek-chat"):
        self.model = model
        self.api_key = settings.DEEPSEEK_API_KEY
        self.api_base = settings.DEEPSEEK_API_BASE

        if not self.api_key or self.api_key == "your_deepseek_api_key":
            logger.warning("[LLMService] DEEPSEEK_API_KEY 未配置，LLM 调用将失败")

        logger.info(f"[LLMService] 初始化完成 | model={model}")

    def _call_llm(self, prompt: str, system_prompt: str = "") -> str:
        """调用 LLM 获取回复"""
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"[LLMService] API 调用失败: {type(e).__name__}: {e}")
            raise

    def analyze_content(self, content: str) -> Dict[str, Any]:
        """分析文本内容，返回结构化分析结果"""
        system_prompt = """你是一个专业的内容分析助手。
请从以下维度分析用户提供的文本，并以 JSON 格式返回：
- summary: 摘要（100字以内）
- keywords: 关键词数组（最多10个）
- entities: 命名实体数组，每个包含 name 和 type
- sentiment: 情感分值 0.0~1.0
- categories: 分类标签数组

仅返回 JSON，不要包含其他内容。"""

        prompt = f"请分析以下内容：\n\n{content}"

        try:
            raw = self._call_llm(prompt, system_prompt)
            # 尝试从返回中提取 JSON
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("\n", 1)[0]
            result = json.loads(raw)
            return {
                "summary": result.get("summary", ""),
                "keywords": result.get("keywords", []),
                "entities": result.get("entities", []),
                "sentiment": float(result.get("sentiment", 0.5)),
                "categories": result.get("categories", []),
            }
        except Exception as e:
            logger.error(f"[LLMService] 内容分析失败: {e}")
            return {
                "summary": "",
                "keywords": [],
                "entities": [],
                "sentiment": 0.5,
                "categories": [],
            }

    def qa_with_context(self, question: str, context: str) -> str:
        """基于上下文回答问题"""
        system_prompt = "你是一个知识问答助手。请基于提供的上下文信息回答用户问题。如果上下文不足以回答问题，请如实说明。"
        prompt = f"上下文信息：\n{context}\n\n问题：{question}"

        try:
            return self._call_llm(prompt, system_prompt)
        except Exception as e:
            logger.error(f"[LLMService] QA 问答失败: {e}")
            return "抱歉，当前无法获取答案（LLM 服务异常）。"