"""
RAG 知识库引擎

基于 PostgreSQL + Python numpy 余弦相似度的检索增强生成引擎。
使用阿里云百炼 text-embedding-v4 进行文本向量化。
支持向量搜索、关键词搜索、混合搜索，按博主/标签/时间范围过滤。

所有对外暴露的方法均使用 @tool 注解，供智能体调用。
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain.tools import tool
from pydantic import BaseModel, Field
from sqlalchemy import and_, or_

from database.models import KnowledgeBase
from database.session import SessionLocal
from langchain_community.embeddings import DashScopeEmbeddings
from utils.config import settings
from utils.logger import logger




# ============ Pydantic 数据模型 ============


class DocumentChunk(BaseModel):
    """知识库文档分块"""

    doc_id: int = Field(default=0, description="数据库主键ID")
    video_id: str = Field(default="", description="关联的视频ID")
    content: str = Field(default="", description="文本内容")
    title: str = Field(default="", description="文档标题")
    author: str = Field(default="", description="博主昵称")
    tags: List[str] = Field(default_factory=list, description="标签列表")
    source_url: str = Field(default="", description="来源URL")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="扩展元数据")
    created_at: str = Field(default="", description="创建时间（ISO 8601）")


class SearchResult(BaseModel):
    """搜索结果条目"""

    doc_id: int = Field(default=0, description="文档ID")
    video_id: str = Field(default="", description="关联视频ID")
    content: str = Field(default="", description="匹配文本内容")
    title: str = Field(default="", description="文档标题")
    author: str = Field(default="", description="博主昵称")
    tags: List[str] = Field(default_factory=list, description="标签")
    similarity: float = Field(default=0.0, description="语义相似度（0-1）")
    keyword_score: float = Field(default=0.0, description="关键词匹配得分")
    combined_score: float = Field(default=0.0, description="综合得分")
    created_at: str = Field(default="", description="创建时间")


class AddDocumentsInput(BaseModel):
    """add_documents 参数"""

    contents: List[str] = Field(description="文本内容列表")
    video_id: str = Field(default="", description="关联视频ID")
    title: str = Field(default="", description="文档标题")
    author: str = Field(default="", description="博主昵称")
    tags: List[str] = Field(default_factory=list, description="标签列表")
    source_url: str = Field(default="", description="来源URL")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="扩展元数据")


class SearchInput(BaseModel):
    """search 参数"""

    query: str = Field(description="搜索查询文本")
    top_k: int = Field(default=5, description="返回结果数量")
    author: str = Field(default="", description="按博主过滤")
    tags: List[str] = Field(default_factory=list, description="按标签过滤")
    start_time: str = Field(default="", description="起始时间（ISO 8601）")
    end_time: str = Field(default="", description="结束时间（ISO 8601）")
    keyword_weight: float = Field(default=0.3, description="关键词权重（0-1）")


class DeleteInput(BaseModel):
    """delete 参数"""

    video_id: str = Field(default="", description="按视频ID删除")
    doc_id: int = Field(default=0, description="按文档ID删除")


class UpdateInput(BaseModel):
    """update 参数"""

    doc_id: int = Field(description="文档ID")
    content: str = Field(default="", description="新文本内容")
    title: str = Field(default="", description="新标题")
    tags: List[str] = Field(default_factory=list, description="新标签列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="新扩展元数据")


# ============ 内置中文停用词（精简版） ============

_STOP_WORDS: set = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "什么", "怎么", "如何", "为什么", "因为", "所以", "但是", "虽然", "可以",
    "这个", "那个", "还是", "只是", "已经", "还", "又", "能", "让", "被",
    "把", "从", "与", "或", "等", "及", "向", "对", "为", "以", "之",
    "而", "且", "但", "如", "若", "则", "按", "照", "用", "给", "跟",
    "吗", "呢", "吧", "啊", "哦", "嗯", "哈", "呀", "嘛", "啦",
}


# ============ RAGEngine 主类 ============


class RAGEngine:
    """RAG 知识库引擎

    基于 PostgreSQL + pgvector 实现文档的向量化存储与检索。
    支持向量相似度搜索、关键词搜索、混合搜索。
    可按博主、标签、时间范围过滤结果。
    """

    # 分块参数
    CHUNK_SIZE: int = 500       # 每块最大字符数
    CHUNK_OVERLAP: int = 50     # 块间重叠字符数

    def __init__(self) -> None:
        self.embedding_service = DashScopeEmbeddings(
            model="text-embedding-v4",
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
        )
        logger.info(
            f"RAGEngine 初始化完成 | "
            f"分块大小: {self.CHUNK_SIZE} | 重叠: {self.CHUNK_OVERLAP}"
        )

    # ---------- 内部分块逻辑 ----------

    @staticmethod
    def _split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """按句子边界递归分块

        优先在句号、换行等自然边界处切分，避免在词中间截断。
        """
        if not text or not text.strip():
            return []

        text = text.strip()
        if len(text) <= chunk_size:
            return [text]

        separators = ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "; "]

        chunks: List[str] = []
        _split_recursive(text, separators, chunk_size, overlap, chunks)

        # 合并过短的尾部
        if len(chunks) >= 2 and len(chunks[-1]) < 50:
            chunks[-2] = chunks[-2] + chunks[-1]
            chunks.pop()

        return chunks

    # ---------- 关键词提取 ----------

    @staticmethod
    def _extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """从文本中提取关键词（简易TF算法）"""
        if not text:
            return []

        # 分词：按非中文字符和停用词分割
        words = re.findall(r"[\u4e00-\u9fff\w]+", text.lower())

        # 统计词频
        freq: Dict[str, int] = {}
        for w in words:
            if len(w) < 2 or w in _STOP_WORDS or w.isdigit():
                continue
            freq[w] = freq.get(w, 0) + 1

        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:max_keywords]]

    # ---------- 数据库操作 ----------

    def _get_session(self):
        """获取数据库会话"""
        db = SessionLocal()
        return db

    # ========== 对外暴露的方法（内部实现） ==========

    def add_documents_impl(
        self,
        contents: List[str],
        video_id: str = "",
        title: str = "",
        author: str = "",
        tags: Optional[List[str]] = None,
        source_url: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """添加文档到知识库

        自动分块、向量化、存入 PostgreSQL。

        Args:
            contents: 文本内容列表
            video_id: 关联视频ID
            title: 文档标题
            author: 博主昵称
            tags: 标签列表
            source_url: 来源URL
            metadata: 扩展元数据

        Returns:
            成功插入的条数
        """
        logger.info(
            f"[RAG] 添加文档 | video_id={video_id} | "
            f"内容数={len(contents)} | author={author}"
        )

        tags = tags or []
        metadata = metadata or {}

        all_chunks: List[Tuple[str, int]] = []
        for content in contents:
            chunks = self._split_text(content, self.CHUNK_SIZE, self.CHUNK_OVERLAP)
            for chunk in chunks:
                all_chunks.append((chunk, len(all_chunks)))

        if not all_chunks:
            logger.warning("[RAG] 无有效内容可添加")
            return 0

        logger.info(f"[RAG] 文本已分为 {len(all_chunks)} 个块")

        # 批量嵌入
        texts = [c[0] for c in all_chunks]
        embeddings = self.embedding_service.embed_documents(texts)

        # 存入数据库
        db = self._get_session()
        inserted = 0
        try:
            for i, (chunk_text, chunk_idx) in enumerate(all_chunks):
                emb = embeddings[i]
                if emb is None:
                    logger.warning(f"[RAG] 块 {chunk_idx} 嵌入失败，跳过")
                    continue

                entry = KnowledgeBase(
                    video_id=video_id or None,
                    content=chunk_text,
                    embedding=json.dumps(emb),  # 存为 JSON 字符串，Python 层做余弦
                    title=title,
                    author=author,
                    tags=tags,
                    source_url=source_url,
                    metadata_json=metadata,
                )
                db.add(entry)
                inserted += 1

            db.commit()
            logger.info(f"[RAG] 添加完成 | 共插入 {inserted} 条记录")

        except Exception as e:
            db.rollback()
            logger.error(f"[RAG] 添加文档失败: {type(e).__name__}: {e}")
            raise
        finally:
            db.close()

        return inserted

    def search_impl(
        self,
        query: str,
        top_k: int = 5,
        author: str = "",
        tags: Optional[List[str]] = None,
        start_time: str = "",
        end_time: str = "",
        keyword_weight: float = 0.3,
    ) -> List[SearchResult]:
        """混合搜索（向量 + 关键词）

        Args:
            query: 搜索查询文本
            top_k: 返回结果数量
            author: 按博主过滤
            tags: 按标签过滤（满足任一即匹配）
            start_time: 起始时间（ISO 8601）
            end_time: 结束时间（ISO 8601）
            keyword_weight: 关键词权重（0-1，0=纯向量，1=纯关键词）

        Returns:
            搜索结果列表，按综合得分降序
        """
        logger.info(
            f"[RAG] 搜索 | query='{query[:50]}...' | top_k={top_k} | "
            f"author='{author}' | tags={tags} | "
            f"time={start_time}~{end_time} | kw_weight={keyword_weight}"
        )

        top_k = max(1, min(top_k, 50))
        keyword_weight = max(0.0, min(keyword_weight, 1.0))
        tags = tags or []

        # 查询向量
        query_embedding = self.embedding_service.embed_query(query)
        if not query_embedding:
            logger.error("[RAG] 查询嵌入失败")
            return []

        # 提取查询中的关键词
        query_keywords = self._extract_keywords(query, max_keywords=5)
        logger.debug(f"[RAG] 查询关键词: {query_keywords}")

        db = self._get_session()
        try:
            # 构建过滤条件
            filters = []
            if author:
                filters.append(KnowledgeBase.author == author)

            if tags:
                # JSON 数组包含任一标签：使用 cast + ilike 兼容性更好的方式
                tag_conditions = [
                    KnowledgeBase.tags.cast(String).ilike(f"%{tag}%")
                    for tag in tags
                ]
                filters.append(or_(*tag_conditions))

            if start_time:
                try:
                    st = datetime.fromisoformat(start_time)
                    filters.append(KnowledgeBase.created_at >= st)
                except ValueError:
                    logger.warning(f"[RAG] 无效的起始时间: {start_time}")

            if end_time:
                try:
                    et = datetime.fromisoformat(end_time)
                    filters.append(KnowledgeBase.created_at <= et)
                except ValueError:
                    logger.warning(f"[RAG] 无效的结束时间: {end_time}")

            # 加载所有匹配过滤条件的条目
            base_query = db.query(KnowledgeBase)
            if filters:
                base_query = base_query.filter(and_(*filters))

            all_rows = base_query.all()
            if not all_rows:
                logger.info("[RAG] 无匹配结果")
                return []

            # Python 层余弦相似度（避免 pgvector 扩展依赖）
            query_vec = np.array(query_embedding, dtype=np.float32)
            query_norm = np.linalg.norm(query_vec)
            if query_norm == 0:
                logger.error("[RAG] 查询向量为零向量")
                return []

            candidates: List[Tuple[Any, float]] = []  # (entry, cosine_similarity)
            for entry in all_rows:
                if not entry.embedding:
                    continue
                try:
                    emb_list = json.loads(entry.embedding)
                    emb_vec = np.array(emb_list, dtype=np.float32)
                    emb_norm = np.linalg.norm(emb_vec)
                    if emb_norm == 0:
                        continue
                    similarity = float(np.dot(query_vec, emb_vec) / (query_norm * emb_norm))
                    candidates.append((entry, similarity))
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

            # 取 top_k * 3 个候选，再进行关键词精排
            candidates.sort(key=lambda x: x[1], reverse=True)
            candidate_count = min(top_k * 3, len(candidates))
            candidates = candidates[:candidate_count]

            if not candidates:
                logger.info("[RAG] 无匹配结果")
                return []

            logger.debug(f"[RAG] 候选结果: {len(candidates)} 条")

            # 混合打分
            results: List[SearchResult] = []
            for entry, sim in candidates:
                vec_similarity = float(sim)

                # 关键词匹配得分
                kw_score = 0.0
                content_lower = entry.content.lower()
                for kw in query_keywords:
                    if kw.lower() in content_lower:
                        kw_score += 1.0 / len(query_keywords) if query_keywords else 0

                combined = (
                    (1.0 - keyword_weight) * vec_similarity
                    + keyword_weight * kw_score
                )

                results.append(
                    SearchResult(
                        doc_id=entry.id,
                        video_id=entry.video_id or "",
                        content=entry.content,
                        title=entry.title or "",
                        author=entry.author or "",
                        tags=entry.tags or [],
                        similarity=round(vec_similarity, 4),
                        keyword_score=round(kw_score, 4),
                        combined_score=round(combined, 4),
                        created_at=entry.created_at.isoformat() if entry.created_at else "",
                    )
                )

            # 按综合得分降序排列
            results.sort(key=lambda x: x.combined_score, reverse=True)
            results = results[:top_k]

            logger.info(f"[RAG] 搜索完成 | 返回 {len(results)} 条")
            return results

        except Exception as e:
            logger.error(f"[RAG] 搜索失败: {type(e).__name__}: {e}")
            return []
        finally:
            db.close()

    def search_vector_impl(
        self,
        query: str,
        top_k: int = 5,
        author: str = "",
        tags: Optional[List[str]] = None,
        start_time: str = "",
        end_time: str = "",
    ) -> List[SearchResult]:
        """纯向量相似度搜索"""
        return self.search_impl(
            query=query,
            top_k=top_k,
            author=author,
            tags=tags,
            start_time=start_time,
            end_time=end_time,
            keyword_weight=0.0,
        )

    def search_keyword_impl(
        self,
        query: str,
        top_k: int = 5,
        author: str = "",
        tags: Optional[List[str]] = None,
        start_time: str = "",
        end_time: str = "",
    ) -> List[SearchResult]:
        """纯关键词搜索（基于 PostgreSQL ILIKE）"""
        logger.info(f"[RAG] 关键词搜索 | query='{query[:50]}...' | top_k={top_k}")

        top_k = max(1, min(top_k, 50))
        tags = tags or []

        db = self._get_session()
        try:
            filters = []
            if author:
                filters.append(KnowledgeBase.author == author)
            if tags:
                tag_conditions = [
                    KnowledgeBase.tags.cast(String).ilike(f"%{tag}%")
                    for tag in tags
                ]
                filters.append(or_(*tag_conditions))
            if start_time:
                try:
                    filters.append(KnowledgeBase.created_at >= datetime.fromisoformat(start_time))
                except ValueError:
                    pass
            if end_time:
                try:
                    filters.append(KnowledgeBase.created_at <= datetime.fromisoformat(end_time))
                except ValueError:
                    pass

            # 关键词 ilike 匹配
            keywords = self._extract_keywords(query, max_keywords=5)
            if not keywords:
                keywords = [query]

            kw_conditions = [
                KnowledgeBase.content.ilike(f"%{kw}%") for kw in keywords
            ]
            filters.append(or_(*kw_conditions))

            base_query = db.query(KnowledgeBase)
            if filters:
                base_query = base_query.filter(and_(*filters))

            rows = base_query.limit(top_k).all()

            results = [
                SearchResult(
                    doc_id=r.id,
                    video_id=r.video_id or "",
                    content=r.content,
                    title=r.title or "",
                    author=r.author or "",
                    tags=r.tags or [],
                    keyword_score=1.0,
                    combined_score=1.0,
                    created_at=r.created_at.isoformat() if r.created_at else "",
                )
                for r in rows
            ]

            logger.info(f"[RAG] 关键词搜索完成 | 返回 {len(results)} 条")
            return results

        except Exception as e:
            logger.error(f"[RAG] 关键词搜索失败: {type(e).__name__}: {e}")
            return []
        finally:
            db.close()

    def delete_impl(self, video_id: str = "", doc_id: int = 0) -> int:
        """删除知识库中的文档

        按 video_id（删除该视频所有文档）或 doc_id（删除单条记录）删除。

        Args:
            video_id: 按视频ID删除所有文档
            doc_id: 按文档ID删除单条

        Returns:
            删除的记录数
        """
        logger.info(f"[RAG] 删除 | video_id={video_id} | doc_id={doc_id}")

        db = self._get_session()
        try:
            if doc_id > 0:
                deleted = db.query(KnowledgeBase).filter(
                    KnowledgeBase.id == doc_id
                ).delete()
                db.commit()
                logger.info(f"[RAG] 已删除 {deleted} 条记录 (doc_id={doc_id})")
                return deleted

            elif video_id:
                deleted = db.query(KnowledgeBase).filter(
                    KnowledgeBase.video_id == video_id
                ).delete()
                db.commit()
                logger.info(f"[RAG] 已删除 {deleted} 条记录 (video_id={video_id})")
                return deleted

            else:
                logger.warning("[RAG] 删除: 未指定 video_id 或 doc_id")
                return 0

        except Exception as e:
            db.rollback()
            logger.error(f"[RAG] 删除失败: {type(e).__name__}: {e}")
            raise
        finally:
            db.close()

    def update_impl(
        self,
        doc_id: int,
        content: str = "",
        title: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """更新文档内容和向量

        更新文本后会重新计算嵌入向量。

        Args:
            doc_id: 文档ID
            content: 新文本内容
            title: 新标题
            tags: 新标签列表
            metadata: 新扩展元数据

        Returns:
            是否更新成功
        """
        logger.info(f"[RAG] 更新 | doc_id={doc_id}")

        db = self._get_session()
        try:
            entry = db.query(KnowledgeBase).filter(KnowledgeBase.id == doc_id).first()
            if not entry:
                logger.warning(f"[RAG] 文档不存在: doc_id={doc_id}")
                return False

            if content:
                entry.content = content
                # 重新嵌入
                new_emb = self.embedding_service.embed_query(content)
                if new_emb:
                    entry.embedding = new_emb

            if title:
                entry.title = title

            if tags is not None:
                entry.tags = tags

            if metadata is not None:
                entry.metadata_json = metadata

            entry.updated_at = datetime.now()
            db.commit()

            logger.info(f"[RAG] 更新成功 | doc_id={doc_id}")
            return True

        except Exception as e:
            db.rollback()
            logger.error(f"[RAG] 更新失败: {type(e).__name__}: {e}")
            return False
        finally:
            db.close()

    def get_stats_impl(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        db = self._get_session()
        try:
            total = db.query(func.count(KnowledgeBase.id)).scalar() or 0
            authors = (
                db.query(func.count(func.distinct(KnowledgeBase.author)))
                .filter(KnowledgeBase.author != "")
                .scalar()
                or 0
            )
            videos = (
                db.query(func.count(func.distinct(KnowledgeBase.video_id)))
                .filter(KnowledgeBase.video_id.isnot(None))
                .scalar()
                or 0
            )
            return {
                "total_documents": total,
                "unique_authors": authors,
                "unique_videos": videos,
            }
        except Exception as e:
            logger.error(f"[RAG] 统计失败: {e}")
            return {"total_documents": 0, "unique_authors": 0, "unique_videos": 0}
        finally:
            db.close()


# ============ 辅助函数 ============

from typing import List as ListType  # noqa: E402


def _split_recursive(
    text: str,
    separators: List[str],
    chunk_size: int,
    overlap: int,
    result: List[str],
) -> None:
    """递归分块辅助函数"""
    if not text:
        return

    if len(text) <= chunk_size:
        result.append(text)
        return

    # 尝试用分隔符切分
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            current = ""
            for part in parts:
                if len(current) + len(sep) + len(part) <= chunk_size:
                    current = (current + sep + part) if current else part
                else:
                    if current:
                        result.append(current)
                    # 重叠部分
                    if overlap > 0 and current:
                        overlap_text = current[-overlap:] if len(current) > overlap else current
                        current = overlap_text + sep + part
                    else:
                        current = part
            if current:
                result.append(current)
            return

    # 无法用分隔符切分，强制按字符数切分
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:
            result.append(chunk)


# ============ LangChain @tool 工具函数 ============

_rag_singleton: Optional[RAGEngine] = None


def _get_rag() -> RAGEngine:
    """获取 RAGEngine 单例"""
    global _rag_singleton
    if _rag_singleton is None:
        _rag_singleton = RAGEngine()
    return _rag_singleton


@tool
def add_documents(
    contents: List[str],
    video_id: str = "",
    title: str = "",
    author: str = "",
    tags: List[str] = None,
    source_url: str = "",
    metadata: Dict[str, Any] = None,
) -> str:
    """将文本内容分块、向量化并存入知识库。

    自动按句子边界分块，每块最多500字符，块间重叠50字符。

    Args:
        contents: 文本内容列表，每条内容会单独分块
        video_id: 关联的抖音视频ID（可选）
        title: 文档标题（可选）
        author: 博主昵称（可选）
        tags: 标签列表，如 ["美食", "教程"]（可选）
        source_url: 来源URL（可选）
        metadata: 扩展元数据字典（可选）

    Returns:
        JSON字符串，格式: {"success": bool, "inserted": int, "error": str|null}
    """
    try:
        if tags is None:
            tags = []
        if metadata is None:
            metadata = {}

        rag = _get_rag()
        inserted = rag.add_documents_impl(
            contents=contents,
            video_id=video_id,
            title=title,
            author=author,
            tags=tags,
            source_url=source_url,
            metadata=metadata,
        )
        return json.dumps(
            {"success": True, "inserted": inserted, "error": None},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"[tool:add_documents] 执行失败: {type(e).__name__}: {e}")
        return json.dumps(
            {"success": False, "inserted": 0, "error": str(e)},
            ensure_ascii=False,
        )


@tool
def search_knowledge(
    query: str,
    top_k: int = 5,
    author: str = "",
    tags: List[str] = None,
    start_time: str = "",
    end_time: str = "",
    keyword_weight: float = 0.3,
) -> str:
    """在知识库中执行混合搜索（向量 + 关键词）。

    同时考虑语义相似度和关键词匹配，支持按博主、标签、时间范围过滤。

    Args:
        query: 搜索查询文本
        top_k: 返回结果数量，默认5，最多50
        author: 按博主昵称过滤（可选，精确匹配）
        tags: 按标签过滤，满足任一标签即匹配（可选）
        start_time: 起始时间，ISO 8601格式（可选）
        end_time: 结束时间，ISO 8601格式（可选）
        keyword_weight: 关键词权重 0-1，默认0.3（0=纯向量，1=纯关键词）

    Returns:
        JSON字符串，格式: {"success": bool, "data": [...], "count": int, "error": str|null}
    """
    try:
        if tags is None:
            tags = []

        rag = _get_rag()
        results = rag.search_impl(
            query=query,
            top_k=top_k,
            author=author,
            tags=tags,
            start_time=start_time,
            end_time=end_time,
            keyword_weight=keyword_weight,
        )
        data = [r.model_dump() for r in results]
        return json.dumps(
            {"success": True, "data": data, "count": len(data), "error": None},
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        logger.error(f"[tool:search_knowledge] 执行失败: {type(e).__name__}: {e}")
        return json.dumps(
            {"success": False, "data": [], "count": 0, "error": str(e)},
            ensure_ascii=False,
        )


@tool
def delete_knowledge(video_id: str = "", doc_id: int = 0) -> str:
    """从知识库中删除文档。

    可指定 video_id（删除该视频的所有关联文档）或 doc_id（删除单条记录）。

    Args:
        video_id: 按视频ID批量删除（可选）
        doc_id: 按文档ID删除单条（可选）

    Returns:
        JSON字符串，格式: {"success": bool, "deleted": int, "video_id": str, "doc_id": int, "error": str|null}
    """
    try:
        rag = _get_rag()
        deleted = rag.delete_impl(video_id=video_id, doc_id=doc_id)
        return json.dumps(
            {
                "success": deleted > 0,
                "deleted": deleted,
                "video_id": video_id,
                "doc_id": doc_id,
                "error": None,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"[tool:delete_knowledge] 执行失败: {type(e).__name__}: {e}")
        return json.dumps(
            {
                "success": False,
                "deleted": 0,
                "video_id": video_id,
                "doc_id": doc_id,
                "error": str(e),
            },
            ensure_ascii=False,
        )


@tool
def update_knowledge(
    doc_id: int,
    content: str = "",
    title: str = "",
    tags: List[str] = None,
    metadata: Dict[str, Any] = None,
) -> str:
    """更新知识库中的文档内容和向量。

    更新文本内容后会自动重新计算嵌入向量。

    Args:
        doc_id: 要更新的文档ID（必填）
        content: 新的文本内容（可选）
        title: 新的标题（可选）
        tags: 新的标签列表（可选）
        metadata: 新的扩展元数据（可选）

    Returns:
        JSON字符串，格式: {"success": bool, "doc_id": int, "error": str|null}
    """
    try:
        if tags is None:
            tags = []
        if metadata is None:
            metadata = {}

        rag = _get_rag()
        ok = rag.update_impl(
            doc_id=doc_id,
            content=content,
            title=title,
            tags=tags,
            metadata=metadata,
        )
        return json.dumps(
            {"success": ok, "doc_id": doc_id, "error": None},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"[tool:update_knowledge] 执行失败: {type(e).__name__}: {e}")
        return json.dumps(
            {"success": False, "doc_id": doc_id, "error": str(e)},
            ensure_ascii=False,
        )


@tool
def get_rag_stats() -> str:
    """获取知识库统计信息。

    Returns:
        JSON字符串，包含 total_documents, unique_authors, unique_videos
    """
    try:
        rag = _get_rag()
        stats = rag.get_stats_impl()
        return json.dumps(stats, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[tool:get_rag_stats] 执行失败: {type(e).__name__}: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)