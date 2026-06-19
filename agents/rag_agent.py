"""
RAG 知识库智能体

负责知识库的管理和问答。
使用 LangGraph create_react_agent 构建。
"""

from langgraph.prebuilt import create_react_agent

from tools.rag_engine import (
    search_knowledge,
    add_documents,
    delete_knowledge,
    get_rag_stats,
    update_knowledge,
)
from utils.config import get_postgres_url, get_lightweight_model, settings
from utils.logger import logger
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg

# 聊天模型
_model = get_lightweight_model()

# ============ 系统提示词 ============

RAG_SYSTEM_PROMPT = """你是一位知识库管理员和问答专家，负责管理视频内容知识库，并回答用户关于视频内容的问题。

## 检索方法
1. 分析用户的问题，提取核心关键词和意图
2. 使用 search_knowledge 工具进行语义搜索，支持向量相似度 + 关键词混合检索
3. 可根据需要按博主（author）、标签（tags）、时间范围（start_time/end_time）过滤
4. 检索结果包含相似度分数，择优选取最相关的 3-5 条结果

## 信息整合
1. 将检索到的多个文档片段按照逻辑关系组织
2. 综合各片段信息，形成完整、准确的回答
3. 如果检索结果不足，应明确告知用户信息局限性
4. 对于有多个来源的信息，进行交叉验证，避免矛盾

## 回答格式
你的回答应包含以下部分：
1. **回答正文**：基于检索结果的完整答案
2. **引用来源**：在正文中标注引用的文档编号，如 [1][2]
3. **来源列表**：回答末尾列出所有引用来源的详细信息，格式为：
   > [1] 标题：xxx | 博主：xxx | 相似度：0.xx | 时间：xxx
   > [2] 标题：xxx | 博主：xxx | 相似度：0.xx | 时间：xxx

## 知识库管理
- 你还可以执行添加、删除、更新知识库文档等管理操作
- 添加文档时，确保内容经过清洗和结构化处理
- 删除文档前，请先确认用户意图，避免误删
- 可以查询知识库统计信息了解数据概况
"""


# 添加记忆检查点以支持多轮对话
def _create_checkpointer():
    """创建 PostgreSQL 检查点（支持多轮对话持久化）"""
    try:
        pg_url = get_postgres_url()  # 例如 postgresql+pg8000://user:pass@host:port/db
        # 移除 SQLAlchemy dialect 前缀, 保留原生 postgresql://
        if "+" in pg_url:
            pg_url = "postgresql://" + pg_url.split("://", 1)[1]
        conn = psycopg.connect(pg_url)
        return PostgresSaver(conn)
    except Exception as e:
        logger.warning(f"PostgresSaver 创建失败（将在无记忆模式下运行）: {e}")
        return None


_memory = _create_checkpointer()

rag_agent = create_react_agent(
    model=_model,
    prompt=RAG_SYSTEM_PROMPT,
    tools=[search_knowledge, add_documents, delete_knowledge, update_knowledge, get_rag_stats],
    checkpointer=_memory,
)

if __name__ == "__main__":
    print("RAGAgent 初始化完成，工具列表:")
    print("  - search_knowledge: 语义搜索知识库")
    print("  - add_documents: 添加文档")
    print("  - delete_knowledge: 删除知识库")
    print("  - update_knowledge: 更新知识库")
    print("  - get_rag_stats: 获取统计信息")