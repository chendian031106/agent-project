from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from api.schemas.request import QueryRequest
from api.schemas.response import QueryResult, ApiResponse
from database.session import get_db
from database.models import Video, VideoContent
from database.vector_store import VectorStore
from tools.llm_service import LLMService
from utils.logger import logger

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

@router.post("/query", response_model=QueryResult)
async def query_knowledge(request: QueryRequest):
    vector_store = VectorStore()
    results = vector_store.search(
        request.question,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold
    )
    
    if not results:
        return QueryResult(
            answer="未找到相关信息",
            sources=[],
            confidence=0.0
        )
    
    context = "\n".join([r['content'] for r in results])
    llm = LLMService()
    answer = llm.qa_with_context(request.question, context)
    
    sources = [
        {"video_id": r['video_id'], "similarity": r['similarity']}
        for r in results
    ]
    
    avg_confidence = sum(r['similarity'] for r in results) / len(results)
    
    return QueryResult(
        answer=answer,
        sources=sources,
        confidence=avg_confidence
    )

@router.post("/ingest/{video_id}", response_model=ApiResponse)
async def ingest_video(video_id: str, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="视频不存在")
    
    content = db.query(VideoContent).filter(VideoContent.video_id == video_id).first()
    if not content:
        raise HTTPException(status_code=400, detail="视频内容未提取")
    
    all_content = "\n".join([
        c for c in [content.audio_text, content.ocr_text, content.subtitle_text] if c
    ])
    
    if not all_content:
        raise HTTPException(status_code=400, detail="没有可入库的内容")
    
    vector_store = VectorStore()
    vector_store.add_document(video_id, all_content)
    
    logger.info(f"视频 {video_id} 已加入知识库")
    return {"success": True, "message": "内容已加入知识库"}

@router.delete("/ingest/{video_id}", response_model=ApiResponse)
async def remove_from_knowledge(video_id: str):
    vector_store = VectorStore()
    vector_store.delete_by_video_id(video_id)
    return {"success": True, "message": "内容已从知识库移除"}

@router.get("/stats", response_model=dict)
async def get_knowledge_stats():
    vector_store = VectorStore()
    documents = vector_store.get_all_documents()
    return {
        "document_count": len(documents),
        "total_size": sum(len(doc.content) for doc in documents)
    }