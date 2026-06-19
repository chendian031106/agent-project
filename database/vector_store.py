import numpy as np
from sqlalchemy import func, select
from database.models import KnowledgeBase
from database.session import get_db
from langchain_community.embeddings import DashScopeEmbeddings
from utils.logger import logger
from typing import List, Dict

class VectorStore:
    def __init__(self):
        self.embedding_service = DashScopeEmbeddings(
            model="text-embedding-v4",
        )
    
    def add_document(self, video_id: str, content: str):
        embedding = self.embedding_service.embed_text(content)
        if not embedding:
            logger.warning(f"嵌入生成失败，跳过文档: {video_id}")
            return
        
        db = next(get_db())
        try:
            kb_entry = KnowledgeBase(
                video_id=video_id,
                content=content,
                embedding=embedding
            )
            db.add(kb_entry)
            db.commit()
            logger.info(f"文档已添加到知识库: {video_id}")
        except Exception as e:
            db.rollback()
            logger.error(f"添加文档失败: {e}")
            raise
        finally:
            db.close()
    
    def add_documents(self, video_ids: List[str], contents: List[str]):
        for video_id, content in zip(video_ids, contents):
            self.add_document(video_id, content)
    
    def search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.7) -> List[Dict]:
        query_embedding = self.embedding_service.embed_text(query)
        if not query_embedding:
            return []
        
        db = next(get_db())
        try:
            results = db.query(
                KnowledgeBase,
                func.cosine_distance(KnowledgeBase.embedding, query_embedding).label('distance')
            ).order_by('distance').limit(top_k).all()
            
            results_list = []
            for kb_entry, distance in results:
                similarity = 1 - float(distance)
                if similarity >= similarity_threshold:
                    results_list.append({
                        'video_id': kb_entry.video_id,
                        'content': kb_entry.content,
                        'similarity': similarity,
                        'created_at': kb_entry.created_at
                    })
            return results_list
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
        finally:
            db.close()
    
    def delete_by_video_id(self, video_id: str):
        db = next(get_db())
        try:
            db.query(KnowledgeBase).filter(KnowledgeBase.video_id == video_id).delete()
            db.commit()
            logger.info(f"已删除视频 {video_id} 的知识库条目")
        except Exception as e:
            db.rollback()
            logger.error(f"删除文档失败: {e}")
            raise
        finally:
            db.close()
    
    def get_all_documents(self) -> List[KnowledgeBase]:
        db = next(get_db())
        try:
            return db.query(KnowledgeBase).all()
        finally:
            db.close()