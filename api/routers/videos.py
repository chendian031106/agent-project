from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from api.schemas.request import VideoUpdateRequest
from api.schemas.response import VideoInfo, VideoListResponse, ApiResponse
from database.session import get_db
from database.models import Video
from utils.logger import logger

router = APIRouter(prefix="/videos", tags=["videos"])

@router.get("/", response_model=VideoListResponse)
async def get_videos(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str = None,
    db: Session = Depends(get_db)
):
    query = db.query(Video)
    if status:
        query = query.filter(Video.status == status)
    
    total = query.count()
    videos = query.order_by(Video.crawl_time.desc()).offset((page - 1) * page_size).limit(page_size).all()
    
    return {
        "items": videos,
        "total": total,
        "page": page,
        "page_size": page_size
    }

@router.get("/{video_id}", response_model=VideoInfo)
async def get_video(video_id: str, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="视频不存在")
    return video

@router.put("/{video_id}", response_model=ApiResponse)
async def update_video(
    video_id: str,
    request: VideoUpdateRequest,
    db: Session = Depends(get_db)
):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="视频不存在")
    
    if request.title:
        video.title = request.title
    if request.author:
        video.author = request.author
    if request.status:
        video.status = request.status
    
    db.commit()
    db.refresh(video)
    return {"success": True, "message": "视频信息已更新"}

@router.delete("/{video_id}", response_model=ApiResponse)
async def delete_video(video_id: str, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="视频不存在")
    
    db.delete(video)
    db.commit()
    logger.info(f"已删除视频: {video_id}")
    return {"success": True, "message": "视频已删除"}