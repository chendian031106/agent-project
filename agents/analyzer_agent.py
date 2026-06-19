"""
内容分析智能体

负责对提取的文本内容进行深度分析和结构化总结。
使用 LangGraph create_react_agent 构建。
"""

from langgraph.prebuilt import create_react_agent

from tools.analyzer import (
    summarize_content,
    extract_keywords,
    extract_entities,
    analyze_sentiment,
    categorize_content,
    deep_analyze,
)
from utils.config import get_chat_model
from utils.logger import logger

# 聊天模型（env 配置的大模型）
_model = get_chat_model()

# 分析智能体的系统提示词
ANALYZER_SYSTEM_PROMPT = """你是一位资深的内容分析专家，能够从海量文本中提炼关键信息，生成有价值的洞察报告。

你的能力和约束：
1. 对视频提取的内容进行深度分析和结构化总结
2. 提取关键信息：摘要、关键词、命名实体、情感倾向、分类标签
3. 生成有洞察力的分析报告，而非简单的信息堆砌
4. 支持基于知识库的智能问答

工作时请遵循：
- 仔细阅读和分析所有提供的文本内容
- 确保分析结果结构化、有条理
- 情感分析要客观中立
- 总结要简明扼要但信息完整
"""

analyzer_agent = create_react_agent(
    model=_model,
    prompt=ANALYZER_SYSTEM_PROMPT,
    tools=[summarize_content, extract_keywords, extract_entities, analyze_sentiment, categorize_content, deep_analyze],
)

if __name__ == "__main__":
    print("AnalyzerAgent 初始化完成，工具列表:")
    print("  - summarize_content: 内容总结")
    print("  - extract_keywords: 提取关键词")
    print("  - extract_entities: 提取命名实体")
    print("  - analyze_sentiment: 情感分析")
    print("  - categorize_content: 内容分类")
    print("  - deep_analyze: 深度分析")