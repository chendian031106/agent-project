from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from api.schemas.request import CrawlRequest
from api.schemas.response import CrawlTaskInfo, ApiResponse
from database.session import get_db
from database.models import CrawlTask
from tools.douyin_crawler import DouyinCrawler
from utils.logger import logger
from datetime import datetime, timedelta

router = APIRouter(prefix="/crawler", tags=["crawler"])

@router.post("/start", response_model=ApiResponse)
async def start_crawl(request: CrawlRequest, db: Session = Depends(get_db)):
    crawler = DouyinCrawler()
    results = []
    
    for url in request.urls:
        url_str = str(url)
        existing_task = db.query(CrawlTask).filter(CrawlTask.url == url_str).first()
        
        if existing_task:
            results.append({"url": url_str, "status": "already_exists", "task_id": existing_task.id})
            continue
        
        try:
            result = crawler.download_video(url_str)
            
            task = CrawlTask(
                url=url_str,
                status="completed",
                last_run_at=datetime.now(),
                next_run_at=datetime.now() + timedelta(seconds=request.monitor_interval) if request.auto_monitor else None
            )
            db.add(task)
            db.commit()
            
            results.append({
                "url": url_str,
                "status": "success",
                "task_id": task.id,
                "video_id": result.get("video_id")
            })
            logger.info(f"爬取成功: {url_str}")
        except Exception as e:
            task = CrawlTask(
                url=url_str,
                status="failed",
                error_message=str(e)
            )
            db.add(task)
            db.commit()
            
            results.append({
                "url": url_str,
                "status": "failed",
                "task_id": task.id,
                "error": str(e)
            })
            logger.error(f"爬取失败 {url_str}: {e}")
    
    return {"success": True, "message": "爬取任务完成", "data": {"results": results}}

@router.get("/tasks", response_model=list[CrawlTaskInfo])
async def get_crawl_tasks(status: str = None, db: Session = Depends(get_db)):
    query = db.query(CrawlTask)
    if status:
        query = query.filter(CrawlTask.status == status)
    return query.order_by(CrawlTask.created_at.desc()).all()

@router.get("/tasks/{task_id}", response_model=CrawlTaskInfo)
async def get_crawl_task(task_id: int, db: Session = Depends(get_db)):
    task = db.query(CrawlTask).filter(CrawlTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task

@router.delete("/tasks/{task_id}", response_model=ApiResponse)
async def delete_crawl_task(task_id: int, db: Session = Depends(get_db)):
    task = db.query(CrawlTask).filter(CrawlTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    db.delete(task)
    db.commit()
    return {"success": True, "message": "任务已删除"}