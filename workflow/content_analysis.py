"""
内容分析工作流 — LangGraph StateGraph

流水线：摘要 → 关键词 → 实体 → 情感 → 分类
使用条件边做错误处理，无内容时提前终止。
"""

import json
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from tools.analyzer import (
    analyze_sentiment as _analyze_sentiment_tool,
    categorize_content as _categorize_content_tool,
    extract_entities as _extract_entities_tool,
    extract_keywords as _extract_keywords_tool,
    summarize_content as _summarize_content_tool,
)
from utils.logger import logger


# ============ 状态定义 ============


class AnalysisState(TypedDict):
    """内容分析工作流状态"""

    content: str  # 输入文本

    # 各阶段结果
    summary: Optional[str]
    keywords: Optional[List[str]]
    entities: Optional[List[Dict[str, str]]]
    sentiment: Optional[float]
    sentiment_label: Optional[str]
    categories: Optional[List[str]]

    # 各阶段错误
    summarize_error: Optional[str]
    keywords_error: Optional[str]
    entities_error: Optional[str]
    sentiment_error: Optional[str]
    categorize_error: Optional[str]


# ============ JSON 安全解析 ============


def _safe_parse(raw: str, field: str, fallback=None):
    """安全解析工具返回的 JSON 字符串"""
    try:
        data = json.loads(raw) if isinstance(raw, str) else (raw or {})
        if isinstance(data, dict):
            return data.get(field, fallback)
        return fallback
    except (json.JSONDecodeError, TypeError, AttributeError):
        return fallback


# ============ 节点函数 ============


def _summarize(state: AnalysisState) -> dict:
    """节点1：生成摘要"""
    content = state.get("content", "")
    if not content:
        return {"summary": None, "summarize_error": "内容为空"}

    logger.info(f"[workflow] 开始生成摘要（文本长度: {len(content)} 字符）")
    try:
        raw = _summarize_content_tool.invoke({"content": content, "max_length": 300})
        summary = _safe_parse(raw, "summary", "")
        return {"summary": summary, "summarize_error": None}
    except Exception as e:
        logger.error(f"[workflow] 摘要生成失败: {e}")
        return {"summary": None, "summarize_error": str(e)}


def _extract_keywords(state: AnalysisState) -> dict:
    """节点2：提取关键词"""
    content = state.get("content", "")
    if not content:
        return {"keywords": [], "keywords_error": "内容为空"}

    logger.info("[workflow] 开始提取关键词")
    try:
        raw = _extract_keywords_tool.invoke({"content": content, "count": 10})
        keywords = _safe_parse(raw, "keywords", [])
        return {"keywords": keywords, "keywords_error": None}
    except Exception as e:
        logger.error(f"[workflow] 关键词提取失败: {e}")
        return {"keywords": [], "keywords_error": str(e)}


def _extract_entities(state: AnalysisState) -> dict:
    """节点3：识别命名实体"""
    content = state.get("content", "")
    if not content:
        return {"entities": [], "entities_error": "内容为空"}

    logger.info("[workflow] 开始实体识别")
    try:
        raw = _extract_entities_tool.invoke({"content": content})
        entities = _safe_parse(raw, "entities", [])
        return {"entities": entities, "entities_error": None}
    except Exception as e:
        logger.error(f"[workflow] 实体识别失败: {e}")
        return {"entities": [], "entities_error": str(e)}


def _analyze_sentiment(state: AnalysisState) -> dict:
    """节点4：情感分析"""
    content = state.get("content", "")
    if not content:
        return {"sentiment": 0.0, "sentiment_label": "中性", "sentiment_error": "内容为空"}

    logger.info("[workflow] 开始情感分析")
    try:
        raw = _analyze_sentiment_tool.invoke({"content": content})
        data = json.loads(raw) if isinstance(raw, str) else (raw or {})
        return {
            "sentiment": float(data.get("sentiment", 0.0)),
            "sentiment_label": data.get("label", "中性"),
            "sentiment_error": None,
        }
    except Exception as e:
        logger.error(f"[workflow] 情感分析失败: {e}")
        return {"sentiment": 0.0, "sentiment_label": "中性", "sentiment_error": str(e)}


def _categorize(state: AnalysisState) -> dict:
    """节点5：内容分类"""
    content = state.get("content", "")
    if not content:
        return {"categories": [], "categorize_error": "内容为空"}

    logger.info("[workflow] 开始内容分类")
    try:
        raw = _categorize_content_tool.invoke({"content": content, "count": 5})
        categories = _safe_parse(raw, "categories", [])
        return {"categories": categories, "categorize_error": None}
    except Exception as e:
        logger.error(f"[workflow] 内容分类失败: {e}")
        return {"categories": [], "categorize_error": str(e)}


# ============ 条件路由函数 ============


def _route_after_summarize(state: AnalysisState) -> str:
    """摘要后路由：有内容 → 关键词，无内容 → END"""
    if state.get("summarize_error") and not state.get("summary"):
        return "end"
    return "keywords"


def _route_after_keywords(state: AnalysisState) -> str:
    """关键词后路由：总是继续到实体"""
    return "entities"


def _route_after_entities(state: AnalysisState) -> str:
    """实体后路由：总是继续到情感"""
    return "sentiment"


def _route_after_sentiment(state: AnalysisState) -> str:
    """情感后路由：总是继续到分类"""
    return "categories"


# ============ 工作流类 ============


class ContentAnalysisWorkflow:
    """内容分析工作流

    使用 LangGraph StateGraph 构造文本分析流水线：
    摘要 → 关键词 → 实体识别 → 情感分析 → 内容分类

    同时提供 Q&A 问答的便捷方法。
    """

    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AnalysisState)

        # 注册节点
        workflow.add_node("summarize", _summarize)
        workflow.add_node("keywords", _extract_keywords)
        workflow.add_node("entities", _extract_entities)
        workflow.add_node("sentiment", _analyze_sentiment)
        workflow.add_node("categories", _categorize)

        # START → summarize
        workflow.add_edge(START, "summarize")

        # 条件边：summarize → keywords / END
        workflow.add_conditional_edges(
            "summarize",
            _route_after_summarize,
            {"keywords": "keywords", "end": END},
        )

        # 顺序边
        workflow.add_edge("keywords", "entities")
        workflow.add_edge("entities", "sentiment")
        workflow.add_edge("sentiment", "categories")

        # categories → END
        workflow.add_edge("categories", END)

        return workflow.compile()

    # ---------- 公开接口 ----------

    def analyze(self, content: str) -> dict:
        """对文本内容进行完整分析

        Args:
            content: 需要分析的文本内容

        Returns:
            包含 summary, keywords, entities, sentiment, sentiment_label, categories 的字典
        """
        if not content or not content.strip():
            logger.warning("[workflow] 输入内容为空，跳过分析")
            return {
                "content": "",
                "summary": None,
                "keywords": [],
                "entities": [],
                "sentiment": 0.0,
                "sentiment_label": "中性",
                "categories": [],
            }

        initial_state: AnalysisState = {
            "content": content,
            "summary": None,
            "keywords": None,
            "entities": None,
            "sentiment": None,
            "sentiment_label": None,
            "categories": None,
            "summarize_error": None,
            "keywords_error": None,
            "entities_error": None,
            "sentiment_error": None,
            "categorize_error": None,
        }

        logger.info(f"[workflow] ContentAnalysisWorkflow 开始分析（文本长度: {len(content)} 字符）")
        result = self.graph.invoke(initial_state)
        logger.info("[workflow] ContentAnalysisWorkflow 分析完成")

        return dict(result)

    def qa(self, question: str, context: str) -> str:
        """基于上下文的问答

        使用 LLM 直接在上下文中回答用户问题。
        如需基于知识库的问答，请使用 RAGAgent。

        Args:
            question: 用户问题
            context: 上下文文本

        Returns:
            回答文本
        """
        if not question or not context:
            return ""

        try:
            from agents.analyzer_agent import AnalyzerAgent

            agent = AnalyzerAgent()
            prompt = (
                f"请基于以下上下文回答用户问题。\n"
                f"如果上下文中没有足够信息，请如实告知。\n\n"
                f"上下文：{context}\n\n"
                f"问题：{question}"
            )
            result = agent.run(
                agent.create_agent_input(task=prompt)
            )
            return result.output or ""
        except Exception as e:
            logger.error(f"[workflow] 问答失败: {e}")
            return ""