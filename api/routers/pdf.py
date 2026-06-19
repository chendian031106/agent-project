from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
from fastapi.responses import FileResponse
from api.schemas.request import PDFGenerateRequest
from api.schemas.response import PDFResponse, ApiResponse
from database.session import get_db
from database.models import Video, AnalysisResult, PDFGeneration
from tools.pdf_generator import PDFGenerator
from utils.config import settings
from utils.logger import logger
from datetime import datetime
from pathlib import Path

router = APIRouter(prefix="/pdf", tags=["pdf"])

@router.post("/generate", response_model=PDFResponse)
async def generate_pdf(request: PDFGenerateRequest, db: Session = Depends(get_db)):
    if not request.video_ids:
        raise HTTPException(status_code=400, detail="视频ID列表不能为空")
    
    video_data_list = []
    analysis_results = []
    
    for video_id in request.video_ids:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail=f"视频 {video_id} 不存在")
        
        video_data = {
            "video_id": video.id,
            "title": video.title,
            "author": video.author,
            "duration": video.duration,
            "view_count": video.view_count,
            "like_count": video.like_count,
            "comment_count": video.comment_count
        }
        video_data_list.append(video_data)
        
        analysis = db.query(AnalysisResult).filter(AnalysisResult.video_id == video_id).first()
        if analysis:
            analysis_results.append({
                "summary": analysis.summary,
                "keywords": analysis.keywords,
                "entities": analysis.entities,
                "sentiment": analysis.sentiment,
                "categories": analysis.categories
            })
        else:
            analysis_results.append({})
    
    pdf_generator = PDFGenerator()
    
    if len(request.video_ids) == 1:
        pdf_path = pdf_generator.generate_pdf(video_data_list[0], analysis_results[0])
    else:
        pdf_path = pdf_generator.generate_batch_pdf(video_data_list, analysis_results)
    
    pdf_name = Path(pdf_path).name
    pdf_record = PDFGeneration(
        video_ids=request.video_ids,
        pdf_path=pdf_path,
        generated_at=datetime.now()
    )
    db.add(pdf_record)
    db.commit()
    
    download_url = f"/api/pdf/download/{pdf_record.id}"
    
    logger.info(f"PDF生成成功: {pdf_path}")
    return PDFResponse(
        pdf_id=str(pdf_record.id),
        download_url=download_url,
        generated_at=pdf_record.generated_at
    )

@router.get("/download/{pdf_id}")
async def download_pdf(pdf_id: int, db: Session = Depends(get_db)):
    pdf_record = db.query(PDFGeneration).filter(PDFGeneration.id == pdf_id).first()
    if not pdf_record:
        raise HTTPException(status_code=404, detail="PDF不存在")
    
    pdf_path = Path(pdf_record.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF文件不存在")
    
    return FileResponse(
        path=str(pdf_path),
        filename=pdf_path.name,
        media_type="application/pdf"
    )

@router.get("/history", response_model=list[PDFResponse])
async def get_pdf_history(db: Session = Depends(get_db)):
    records = db.query(PDFGeneration).order_by(PDFGeneration.generated_at.desc()).all()
    return [
        PDFResponse(
            pdf_id=str(record.id),
            download_url=f"/api/pdf/download/{record.id}",
            generated_at=record.generated_at
        )
        for record in records
    ]

@router.delete("/{pdf_id}", response_model=ApiResponse)
async def delete_pdf(pdf_id: int, db: Session = Depends(get_db)):
    pdf_record = db.query(PDFGeneration).filter(PDFGeneration.id == pdf_id).first()
    if not pdf_record:
        raise HTTPException(status_code=404, detail="PDF不存在")
    
    pdf_path = Path(pdf_record.pdf_path)
    if pdf_path.exists():
        pdf_path.unlink()
    
    db.delete(pdf_record)
    db.commit()
    
    return {"success": True, "message": "PDF已删除"}