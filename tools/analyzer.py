"""
内容分析工具集

提供文本分析所需的 @tool 函数，供 AnalyzerAgent 调用。
所有工具均基于 DeepSeek V4 LLM，返回结构化 JSON 字符串。

工具列表：
- summarize_content    : 生成文本摘要
- extract_keywords     : 提取关键词
- extract_entities     : 识别命名实体
- analyze_sentiment    : 情感分析
- categorize_content   : 内容分类
- deep_analyze         : 一键综合深度分析
"""

import json
from typing import Any, Dict, List, Optional

from langchain.tools import tool
from langchain_openai import ChatOpenAI

from utils.config import settings
from utils.logger import logger

# ============ LLM 工厂 ============


def _get_llm() -> ChatOpenAI:
    """获取 DeepSeek V4 LLM 实例"""
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_API_BASE or "https://api.deepseek.com/v1",
        temperature=0.3,  # 分析任务用低温度，确保稳定性
        max_tokens=2048,
    )


def _safe_json_parse(raw: str, fallback: Any = None) -> Any:
    """安全解析 JSON，失败时返回 fallback"""
    try:
        # 尝试提取 ```json ... ``` 代码块
        if "```json" in raw:
            start = raw.index("```json") + 7
            end = raw.index("```", start)
            raw = raw[start:end].strip()
        elif "```" in raw:
            start = raw.index("```") + 3
            end = raw.index("```", start)
            raw = raw[start:end].strip()
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return fallback if fallback is not None else raw


# ============ @tool 工具函数 ============


@tool
def summarize_content(content: str, max_length: int = 300) -> str:
    """对文本内容生成精炼摘要。

    提取核心观点和关键信息，用简洁的语言概括。

    Args:
        content: 需要总结的文本内容
        max_length: 摘要最大长度（字符数），默认300

    Returns:
        JSON字符串，格式: {"success": bool, "summary": str, "error": str|null}
    """
    if not content or not content.strip():
        return json.dumps({"success": False, "summary": "", "error": "输入内容为空"}, ensure_ascii=False)

    try:
        llm = _get_llm()
        prompt = (
            f"请用简洁的语言总结以下文本内容的核心要点，不超过{max_length}字。\n"
            f"只输出摘要文本，不要添加任何前缀或说明。\n\n"
            f"{content[:8000]}"  # 限制输入长度
        )
        response = llm.invoke(prompt)
        summary = response.content.strip() if response.content else ""

        return json.dumps(
            {"success": True, "summary": summary, "error": None},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"[tool:summarize_content] 执行失败: {type(e).__name__}: {e}")
        return json.dumps(
            {"success": False, "summary": "", "error": str(e)},
            ensure_ascii=False,
        )


@tool
def extract_keywords(content: str, count: int = 10) -> str:
    """从文本中提取核心关键词。

    识别文本中最重要的关键词或短语，按重要性排序。

    Args:
        content: 需要提取关键词的文本内容
        count: 提取关键词数量，默认10个

    Returns:
        JSON字符串，格式: {"success": bool, "keywords": [str, ...], "error": str|null}
    """
    if not content or not content.strip():
        return json.dumps({"success": False, "keywords": [], "error": "输入内容为空"}, ensure_ascii=False)

    try:
        llm = _get_llm()
        prompt = (
            f"从以下文本中提取{count}个最重要的关键词或短语，按重要性从高到低排序。\n"
            f"输出格式：纯JSON数组，如 [\"关键词1\", \"关键词2\", ...]\n"
            f"只输出JSON数组，不要添加任何其他文字。\n\n"
            f"{content[:8000]}"
        )
        response = llm.invoke(prompt)
        keywords = _safe_json_parse(response.content, [])
        if not isinstance(keywords, list):
            keywords = []

        return json.dumps(
            {"success": True, "keywords": keywords[:count], "error": None},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"[tool:extract_keywords] 执行失败: {type(e).__name__}: {e}")
        return json.dumps(
            {"success": False, "keywords": [], "error": str(e)},
            ensure_ascii=False,
        )


@tool
def extract_entities(content: str) -> str:
    """从文本中识别命名实体。

    识别人物、地点、组织、产品、时间、事件等命名实体。

    Args:
        content: 需要识别实体的文本内容

    Returns:
        JSON字符串，格式: {"success": bool, "entities": [{"name": str, "type": str}, ...], "error": str|null}
        type 取值: 人物/地点/组织/产品/时间/事件/其他
    """
    if not content or not content.strip():
        return json.dumps({"success": False, "entities": [], "error": "输入内容为空"}, ensure_ascii=False)

    try:
        llm = _get_llm()
        prompt = (
            "从以下文本中识别所有命名实体，包括人物、地点、组织、产品、时间、事件等。\n"
            "输出格式：纯JSON数组，每个元素为 {\"name\": \"实体名称\", \"type\": \"实体类型\"}。\n"
            "实体类型取值：人物/地点/组织/产品/时间/事件/其他\n"
            "只输出JSON数组，不要添加任何其他文字。\n\n"
            f"{content[:8000]}"
        )
        response = llm.invoke(prompt)
        entities = _safe_json_parse(response.content, [])
        if not isinstance(entities, list):
            entities = []

        return json.dumps(
            {"success": True, "entities": entities, "error": None},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"[tool:extract_entities] 执行失败: {type(e).__name__}: {e}")
        return json.dumps(
            {"success": False, "entities": [], "error": str(e)},
            ensure_ascii=False,
        )


@tool
def analyze_sentiment(content: str) -> str:
    """分析文本的情感倾向。

    判断文本的情感倾向，返回分值和标签。

    Args:
        content: 需要分析情感的文本内容

    Returns:
        JSON字符串，格式: {"success": bool, "sentiment": float, "label": str, "confidence": float, "error": str|null}
        sentiment: -1.0(极度负面) ~ 1.0(极度正面)
        label: 正面/负面/中性
        confidence: 置信度 0.0~1.0
    """
    if not content or not content.strip():
        return json.dumps(
            {"success": False, "sentiment": 0.0, "label": "中性", "confidence": 0.0, "error": "输入内容为空"},
            ensure_ascii=False,
        )

    try:
        llm = _get_llm()
        prompt = (
            "分析以下文本的情感倾向。\n"
            "输出格式：纯JSON对象，包含三个字段：\n"
            "- sentiment: 情感分值，-1.0(极度负面)到1.0(极度正面)之间的浮点数\n"
            "- label: 正面/负面/中性\n"
            "- confidence: 置信度 0.0~1.0\n"
            "只输出JSON对象，不要添加任何其他文字。\n\n"
            f"{content[:8000]}"
        )
        response = llm.invoke(prompt)
        result = _safe_json_parse(response.content, {})

        if not isinstance(result, dict):
            result = {}

        return json.dumps(
            {
                "success": True,
                "sentiment": float(result.get("sentiment", 0.0)),
                "label": result.get("label", "中性"),
                "confidence": float(result.get("confidence", 0.5)),
                "error": None,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"[tool:analyze_sentiment] 执行失败: {type(e).__name__}: {e}")
        return json.dumps(
            {
                "success": False,
                "sentiment": 0.0,
                "label": "中性",
                "confidence": 0.0,
                "error": str(e),
            },
            ensure_ascii=False,
        )


@tool
def categorize_content(content: str, count: int = 5) -> str:
    """为文本内容分类打标签。

    识别文本的主题领域，返回分类标签。

    Args:
        content: 需要分类的文本内容
        count: 返回标签数量，默认5个

    Returns:
        JSON字符串，格式: {"success": bool, "categories": [str, ...], "error": str|null}
    """
    if not content or not content.strip():
        return json.dumps({"success": False, "categories": [], "error": "输入内容为空"}, ensure_ascii=False)

    try:
        llm = _get_llm()
        prompt = (
            f"为以下文本内容进行分类，输出{count}个最相关的分类标签。\n"
            "标签应涵盖主题领域，如：科技/美食/旅游/教育/娱乐/体育/财经/生活/时尚/游戏 等。\n"
            "输出格式：纯JSON数组，如 [\"科技\", \"AI\", \"编程\"]\n"
            "只输出JSON数组，不要添加任何其他文字。\n\n"
            f"{content[:8000]}"
        )
        response = llm.invoke(prompt)
        categories = _safe_json_parse(response.content, [])
        if not isinstance(categories, list):
            categories = []

        return json.dumps(
            {"success": True, "categories": categories[:count], "error": None},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"[tool:categorize_content] 执行失败: {type(e).__name__}: {e}")
        return json.dumps(
            {"success": False, "categories": [], "error": str(e)},
            ensure_ascii=False,
        )


@tool
def deep_analyze(content: str) -> str:
    """对文本内容进行综合深度分析（一键完成所有分析）。

    一次调用完成摘要、关键词提取、实体识别、情感分析和分类。
    适合需要完整分析报告的场景。

    Args:
        content: 需要分析的文本内容

    Returns:
        JSON字符串，格式:
        {
            "success": bool,
            "summary": str,
            "keywords": [str, ...],
            "entities": [{"name": str, "type": str}, ...],
            "sentiment": float,
            "sentiment_label": str,
            "categories": [str, ...],
            "error": str|null
        }
    """
    if not content or not content.strip():
        return json.dumps(
            {
                "success": False,
                "summary": "",
                "keywords": [],
                "entities": [],
                "sentiment": 0.0,
                "sentiment_label": "中性",
                "categories": [],
                "error": "输入内容为空",
            },
            ensure_ascii=False,
        )

    try:
        llm = _get_llm()
        prompt = (
            "请对以下文本内容进行全面深度分析，输出一个JSON对象，包含以下字段：\n"
            "1. summary: 内容摘要（200字以内）\n"
            "2. keywords: 关键词列表，5-10个\n"
            "3. entities: 命名实体列表，每个元素为 {\"name\": \"实体名\", \"type\": \"人物/地点/组织/产品/时间/事件\"}\n"
            "4. sentiment: 情感分值，-1.0(极度负面) ~ 1.0(极度正面)\n"
            "5. sentiment_label: 正面/负面/中性\n"
            "6. categories: 分类标签列表，3-5个\n\n"
            "只输出JSON对象，不要添加任何其他文字。\n\n"
            f"{content[:8000]}"
        )
        response = llm.invoke(prompt)
        result = _safe_json_parse(response.content, {})

        if not isinstance(result, dict):
            result = {}

        return json.dumps(
            {
                "success": True,
                "summary": result.get("summary", ""),
                "keywords": result.get("keywords", []),
                "entities": result.get("entities", []),
                "sentiment": float(result.get("sentiment", 0.0)),
                "sentiment_label": result.get("sentiment_label", "中性"),
                "categories": result.get("categories", []),
                "error": None,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"[tool:deep_analyze] 执行失败: {type(e).__name__}: {e}")
        return json.dumps(
            {
                "success": False,
                "summary": "",
                "keywords": [],
                "entities": [],
                "sentiment": 0.0,
                "sentiment_label": "中性",
                "categories": [],
                "error": str(e),
            },
            ensure_ascii=False,
        )