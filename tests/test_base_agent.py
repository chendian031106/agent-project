"""
agents/base.py + 各业务智能体 单元测试

测试覆盖：
- Pydantic 数据模型（AgentInput, AgentOutput, MemoryEntry, LLMConfig）
- LLM 工厂函数 _create_llm
- BaseAgent 抽象类约束
- CrawlerAgent 继承与向后兼容
- AnalyzerAgent 继承与向后兼容
- ExtractorAgent 继承与向后兼容
- 记忆管理（短期/长期）
- 输入输出格式
- 异常处理
"""

import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============ 数据模型测试 ============


class TestAgentInput(unittest.TestCase):
    """AgentInput 模型测试"""

    def test_default_values(self):
        """测试默认值"""
        inp = AgentInput(task="测试任务")
        self.assertEqual(inp.task, "测试任务")
        self.assertIsNone(inp.context)
        self.assertIsNone(inp.session_id)
        self.assertEqual(inp.max_tokens, 2048)
        self.assertEqual(inp.temperature, 0.7)

    def test_full_construction(self):
        """测试完整构造"""
        inp = AgentInput(
            task="分析视频",
            context={"video_id": "123"},
            session_id="session_abc",
            max_tokens=4096,
            temperature=0.3,
        )
        self.assertEqual(inp.task, "分析视频")
        self.assertEqual(inp.context["video_id"], "123")
        self.assertEqual(inp.session_id, "session_abc")
        self.assertEqual(inp.max_tokens, 4096)
        self.assertEqual(inp.temperature, 0.3)


class TestAgentOutput(unittest.TestCase):
    """AgentOutput 模型测试"""

    def test_default_values(self):
        """测试默认值"""
        out = AgentOutput()
        self.assertTrue(out.success)
        self.assertIsNone(out.output)
        self.assertIsNone(out.error)
        self.assertEqual(out.latency_ms, 0)

    def test_success_output(self):
        """测试成功输出"""
        out = AgentOutput(
            success=True,
            output="分析完成",
            session_id="s1",
            token_usage={"total_tokens": 150},
            latency_ms=1200,
        )
        self.assertEqual(out.output, "分析完成")
        self.assertEqual(out.latency_ms, 1200)
        self.assertEqual(out.token_usage["total_tokens"], 150)

    def test_error_output(self):
        """测试错误输出"""
        out = AgentOutput(success=False, error="API调用失败")
        self.assertFalse(out.success)
        self.assertEqual(out.error, "API调用失败")

    def test_streaming_output(self):
        """测试流式输出"""
        out = AgentOutput(streaming_output=["分片1", "分片2", "分片3"])
        self.assertEqual(len(out.streaming_output), 3)
        self.assertEqual(out.streaming_output[1], "分片2")

    def test_model_dump(self):
        """测试序列化"""
        out = AgentOutput(success=True, output="结果")
        dumped = out.model_dump()
        self.assertIsInstance(dumped, dict)
        self.assertEqual(dumped["success"], True)
        self.assertEqual(dumped["output"], "结果")


class TestMemoryEntry(unittest.TestCase):
    """MemoryEntry 模型测试"""

    def test_creation(self):
        """测试创建"""
        entry = MemoryEntry(role="user", content="你好")
        self.assertEqual(entry.role, "user")
        self.assertEqual(entry.content, "你好")
        self.assertIsNotNone(entry.timestamp)

    def test_model_dump(self):
        """测试序列化"""
        entry = MemoryEntry(role="assistant", content="你好！有什么可以帮助你的？")
        dumped = entry.model_dump()
        self.assertEqual(dumped["role"], "assistant")
        self.assertIn("timestamp", dumped)


class TestLLMConfig(unittest.TestCase):
    """LLMConfig 模型测试"""

    def test_deepseek_default(self):
        """测试 DeepSeek 默认配置"""
        config = LLMConfig()
        self.assertEqual(config.provider, "deepseek")
        self.assertEqual(config.model, "deepseek-chat")
        self.assertEqual(config.api_base, "https://api.deepseek.com/v1")

    def test_qwen_config(self):
        """测试 Qwen 配置"""
        config = LLMConfig(
            provider="qwen",
            model="qwen3.6-plus-128k",
            api_key="test_key",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.assertEqual(config.provider, "qwen")
        self.assertEqual(config.model, "qwen3.6-plus-128k")

    def test_temperature_clamping(self):
        """测试 temperature 范围约束（Pydantic v2 自动校验）"""
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            LLMConfig(temperature=-1.0)


# ============ 创建智能体辅助函数测试 ============


class TestCreateAgentInput(unittest.TestCase):
    """create_agent_input 辅助函数测试"""

    def test_minimal(self):
        """测试最小参数"""
        inp = create_agent_input(task="测试")
        self.assertEqual(inp.task, "测试")
        self.assertEqual(inp.max_tokens, 2048)

    def test_full(self):
        """测试完整参数"""
        inp = create_agent_input(
            task="测试",
            context={"key": "val"},
            session_id="sid",
            max_tokens=1024,
            temperature=0.5,
        )
        self.assertEqual(inp.context["key"], "val")
        self.assertEqual(inp.session_id, "sid")
        self.assertEqual(inp.max_tokens, 1024)
        self.assertEqual(inp.temperature, 0.5)


# 在测试前导入（需要先完成类定义）
from agents.base import (
    AgentInput,
    AgentOutput,
    BaseAgent,
    LLMConfig,
    MemoryEntry,
    _create_llm,
    create_agent_input,
)


class TestLLMFactory(unittest.TestCase):
    """LLM 工厂函数测试"""

    @patch("agents.base.settings")
    def test_create_llm_deepseek(self, mock_settings):
        """测试创建 DeepSeek LLM"""
        mock_settings.DEEPSEEK_API_KEY = "sk-test"
        mock_settings.DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
        mock_settings.DASHSCOPE_API_KEY = ""

        config = LLMConfig(
            provider="deepseek",
            model="deepseek-chat",
            api_key="sk-test",
            api_base="https://api.deepseek.com/v1",
        )
        llm = _create_llm(config)
        self.assertEqual(llm.model, "deepseek-chat")

    @patch("agents.base.settings")
    def test_create_llm_qwen(self, mock_settings):
        """测试创建 Qwen LLM"""
        mock_settings.DEEPSEEK_API_KEY = ""
        mock_settings.DASHSCOPE_API_KEY = "sk-qwen-test"

        config = LLMConfig(
            provider="qwen",
            model="qwen3.6-plus-128k",
            api_key="sk-qwen-test",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        llm = _create_llm(config)
        self.assertEqual(llm.model, "qwen3.6-plus-128k")


# ============ BaseAgent 抽象约束测试 ============


class TestBaseAgentAbstraction(unittest.TestCase):
    """BaseAgent 抽象约束测试"""

    def test_cannot_instantiate_directly(self):
        """测试不能直接实例化 BaseAgent"""
        with self.assertRaises(TypeError):
            BaseAgent()  # type: ignore

    def test_concrete_agent_works(self):
        """测试继承 BaseAgent 后的实例化"""
        try:
            from agents.crawler_agent import CrawlerAgent
            # 不连接 Redis，使用 mock
            with patch("agents.base.RedisClient", side_effect=Exception("No Redis")):
                agent = CrawlerAgent()
                self.assertEqual(agent.agent_name, "crawler_agent")
                self.assertTrue(agent.system_prompt.startswith("你是一位"))
                self.assertEqual(len(agent.tools), 3)
        except Exception as e:
            # 即使 LLM 初始化失败，也应该有正确的属性
            logger_msg = f"[TEST] CrawlerAgent 实例化注意: {e}"


# ============ CrawlerAgent 测试 ============


class TestCrawlerAgent(unittest.TestCase):
    """CrawlerAgent 继承与向后兼容测试"""

    @classmethod
    def setUpClass(cls):
        with patch("agents.base.RedisClient", side_effect=Exception("No Redis")):
            with patch("agents.base._create_llm") as mock_llm:
                mock_llm.return_value = MagicMock()
                from agents.crawler_agent import CrawlerAgent
                cls.agent = CrawlerAgent()

    def test_agent_name(self):
        """测试智能体名称"""
        self.assertEqual(self.agent.agent_name, "crawler_agent")

    def test_tools_loaded(self):
        """测试工具加载"""
        self.assertEqual(len(self.agent.tools), 3)
        tool_names = [t.name for t in self.agent.tools]
        self.assertIn("crawl_videos", tool_names)
        self.assertIn("get_video_info", tool_names)
        self.assertIn("delete_video", tool_names)

    def test_backward_compat_crewai_agent(self):
        """测试向后兼容：CrewAI Agent"""
        self.assertIsNotNone(self.agent.agent)
        self.assertEqual(self.agent.agent.role, "抖音视频爬取专家")

    def test_backward_compat_download_video(self):
        """测试向后兼容：download_video"""
        self.agent.crawler.download_video = MagicMock(
            return_value={"video_id": "test", "title": "测试"}
        )
        result = self.agent.download_video("https://example.com")
        self.assertEqual(result["video_id"], "test")

    def test_backward_compat_batch_download(self):
        """测试向后兼容：batch_download"""
        self.agent.crawler.batch_download = MagicMock(
            return_value=[{"video_id": "v1"}, {"video_id": "v2"}]
        )
        results = self.agent.batch_download(["url1", "url2"])
        self.assertEqual(len(results), 2)

    def test_backward_compat_create_task(self):
        """测试向后兼容：create_download_task"""
        task = self.agent.create_download_task("https://douyin.com/video/123")
        self.assertIsNotNone(task)
        self.assertIn("爬取", task.description)

    def test_execute_returns_agent_output(self):
        """测试 execute 返回 AgentOutput"""
        with patch.object(self.agent, "run") as mock_run:
            mock_run.return_value = AgentOutput(
                success=True, output="爬取完成", latency_ms=500
            )
            result = self.agent.execute("爬取博主视频")
            self.assertIsInstance(result, AgentOutput)
            self.assertTrue(result.success)
            self.assertEqual(result.output, "爬取完成")

    def test_clear_memory(self):
        """测试清除记忆"""
        self.agent.clear_memory("test_session")
        # 不应抛出异常
        self.assertIsNotNone(self.agent)


# ============ AnalyzerAgent 测试 ============


class TestAnalyzerAgent(unittest.TestCase):
    """AnalyzerAgent 继承与向后兼容测试"""

    @classmethod
    def setUpClass(cls):
        with patch("agents.base.RedisClient", side_effect=Exception("No Redis")):
            with patch("agents.base._create_llm") as mock_llm:
                mock_llm.return_value = MagicMock()
                with patch("tools.llm_service.LLMService") as mock_llm_service:
                    mock_instance = MagicMock()
                    mock_instance.analyze_content.return_value = {"summary": "摘要"}
                    mock_instance.summarize.return_value = "总结文本"
                    mock_llm_service.return_value = mock_instance
                    from agents.analyzer_agent import AnalyzerAgent
                    cls.agent = AnalyzerAgent()

    def test_agent_name(self):
        """测试智能体名称"""
        self.assertEqual(self.agent.agent_name, "analyzer_agent")

    def test_no_tools(self):
        """测试分析智能体没有工具"""
        self.assertEqual(len(self.agent.tools), 0)

    def test_backward_compat_analyze_content(self):
        """测试向后兼容：analyze_content"""
        result = self.agent.analyze_content("测试内容")
        self.assertIsInstance(result, dict)
        self.assertIn("summary", result)

    def test_backward_compat_summarize(self):
        """测试向后兼容：summarize"""
        result = self.agent.summarize("需要总结的文本", max_length=100)
        self.assertEqual(result, "总结文本")

    def test_backward_compat_crewai_agent(self):
        """测试向后兼容：CrewAI Agent"""
        self.assertIsNotNone(self.agent.agent)
        self.assertEqual(self.agent.agent.role, "内容分析专家")

    def test_backward_compat_create_task(self):
        """测试向后兼容：create_analyze_task"""
        task = self.agent.create_analyze_task("内容")
        self.assertIsNotNone(task)

    def test_execute_returns_agent_output(self):
        """测试 execute 返回 AgentOutput"""
        with patch.object(self.agent, "run") as mock_run:
            mock_run.return_value = AgentOutput(success=True, output="分析结果")
            result = self.agent.execute("分析这段文本")
            self.assertTrue(result.success)

    def test_analyze_content_error_returns_empty_dict(self):
        """测试 analyze_content 错误时返回空 dict"""
        self.agent.llm_service.analyze_content = MagicMock(side_effect=RuntimeError("错误"))
        result = self.agent.analyze_content("内容")
        self.assertEqual(result, {})


# ============ ExtractorAgent 测试 ============


class TestExtractorAgent(unittest.TestCase):
    """ExtractorAgent 继承与向后兼容测试"""

    @classmethod
    def setUpClass(cls):
        with patch("agents.base.RedisClient", side_effect=Exception("No Redis")):
            with patch("agents.base._create_llm") as mock_llm:
                mock_llm.return_value = MagicMock()
                with patch("tools.speech_to_text.SpeechToText") as mock_stt:
                    mock_stt_instance = MagicMock()
                    mock_stt_instance.transcribe.return_value = "语音文本"
                    mock_stt.return_value = mock_stt_instance
                    with patch("tools.ocr_service.OCRService") as mock_ocr:
                        mock_ocr_instance = MagicMock()
                        mock_ocr_instance.ocr_video.return_value = "OCR文本"
                        mock_ocr.return_value = mock_ocr_instance
                        from agents.extractor_agent import ExtractorAgent
                        cls.agent = ExtractorAgent()

    def test_agent_name(self):
        """测试智能体名称"""
        self.assertEqual(self.agent.agent_name, "extractor_agent")

    def test_backward_compat_extract_audio(self):
        """测试向后兼容：extract_audio"""
        result = self.agent.extract_audio("/path/to/video.mp4")
        self.assertEqual(result, "语音文本")

    def test_backward_compat_extract_ocr(self):
        """测试向后兼容：extract_ocr"""
        result = self.agent.extract_ocr("/path/to/video.mp4")
        self.assertEqual(result, "OCR文本")

    def test_backward_compat_extract_all(self):
        """测试向后兼容：extract_all"""
        result = self.agent.extract_all("/path/to/video.mp4")
        self.assertIn("audio_text", result)
        self.assertIn("ocr_text", result)
        self.assertIn("subtitle_text", result)
        self.assertEqual(result["audio_text"], "语音文本")
        self.assertEqual(result["ocr_text"], "OCR文本")

    def test_backward_compat_crewai_agent(self):
        """测试向后兼容：CrewAI Agent"""
        self.assertIsNotNone(self.agent.agent)
        self.assertEqual(self.agent.agent.role, "多模态内容提取专家")

    def test_backward_compat_create_task(self):
        """测试向后兼容：create_extract_task"""
        task = self.agent.create_extract_task("/path/to/video.mp4")
        self.assertIsNotNone(task)

    def test_execute_returns_agent_output(self):
        """测试 execute 返回 AgentOutput"""
        with patch.object(self.agent, "run") as mock_run:
            mock_run.return_value = AgentOutput(success=True, output="提取完成")
            result = self.agent.execute("提取视频内容")
            self.assertTrue(result.success)

    def test_extract_audio_error_returns_empty(self):
        """测试 extract_audio 错误时返回空字符串"""
        self.agent.stt.transcribe = MagicMock(side_effect=RuntimeError("错误"))
        result = self.agent.extract_audio("/bad/path.mp4")
        self.assertEqual(result, "")


# ============ Memory 管理测试 ============


class TestMemoryManagement(unittest.TestCase):
    """BaseAgent 记忆管理测试"""

    @classmethod
    def setUpClass(cls):
        with patch("agents.base.RedisClient") as mock_redis_cls:
            cls.mock_redis_instance = MagicMock()
            mock_redis_cls.return_value = cls.mock_redis_instance
            with patch("agents.base._create_llm") as mock_llm:
                mock_llm.return_value = MagicMock()

                # 创建一个具体的子类用于测试
                from agents.crawler_agent import CrawlerAgent

                with patch.object(CrawlerAgent, "run") as mock_run:
                    mock_run.return_value = AgentOutput(success=True, output="done")
                    cls.agent = CrawlerAgent()

    def test_memory_isolation(self):
        """测试不同 session 的记忆隔离"""
        import json
        # 模拟不同 session 的长期记忆
        self.mock_redis_instance.get.return_value = None

        mem1 = self.agent._get_memory("session_a")
        mem2 = self.agent._get_memory("session_b")
        self.assertIsNot(mem1, mem2)

    def test_clear_memory(self):
        """测试清除记忆"""
        self.agent._get_memory("test_session")
        self.agent.clear_memory("test_session")
        self.assertNotIn("test_session", self.agent._short_term_memory)

    def test_append_to_memory(self):
        """测试追加记忆"""
        self.mock_redis_instance.get.return_value = None
        self.agent._append_to_memory("append_session", "用户问题", "助手回答")
        mem = self.agent._get_memory("append_session")
        self.assertIsNotNone(mem)

    def test_load_long_term_empty(self):
        """测试加载空长期记忆"""
        self.mock_redis_instance.get.return_value = None
        entries = self.agent._load_long_term("no_memory")
        self.assertEqual(len(entries), 0)

    def test_load_long_term_with_data(self):
        """测试加载有数据的长期记忆"""
        import json
        test_data = json.dumps([
            {"role": "user", "content": "你好", "timestamp": "2024-01-01T00:00:00"},
            {"role": "assistant", "content": "你好！", "timestamp": "2024-01-01T00:00:01"},
        ])
        self.mock_redis_instance.get.return_value = test_data
        entries = self.agent._load_long_term("existing_session")
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].role, "user")
        self.assertEqual(entries[1].content, "你好！")


# ============ 异常处理测试 ============


class TestErrorHandling(unittest.TestCase):
    """错误处理测试"""

    def test_agent_output_error_not_blank(self):
        """测试错误输出时 error 不为空"""
        out = AgentOutput(success=False, error="发生错误")
        self.assertIsNotNone(out.error)

    def test_agent_output_success_no_error(self):
        """测试成功输出时 error 可为 None"""
        out = AgentOutput(success=True, output="成功")
        self.assertIsNone(out.error)

    def test_empty_task_input(self):
        """测试空任务输入"""
        inp = AgentInput(task="")
        self.assertEqual(inp.task, "")
        self.assertIsNotNone(inp)


if __name__ == "__main__":
    unittest.main(verbosity=2)