from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from api.schemas.request import AnalysisRequest
from api.schemas.response import AnalysisResult, ContentExtract, ApiResponse
from database.session import get_db
from database.models import Video, VideoContent, AnalysisResult as AnalysisResultModel
from tools.speech_to_text import SpeechToText
# from tools.ocr_service import OCRService  # OCR 暂时注释
from tools.llm_service import LLMService
from utils.logger import logger
from datetime import datetime


router = APIRouter(prefix="/analysis", tags=["analysis"])

@router.post("/video/{video_id}", response_model=AnalysisResult)
async def analyze_video(
    video_id: str,
    request: AnalysisRequest = None,
    db: Session = Depends(get_db)
):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="视频不存在")
    
    if not video.video_path:
        raise HTTPException(status_code=400, detail="视频文件不存在")
    
    req = request or AnalysisRequest(video_id=video_id)
    content_extract = ContentExtract()
    
    if req.extract_audio:
        try:
            stt = SpeechToText()
            content_extract.audio_text = stt.transcribe(video.video_path)
        except Exception as e:
            logger.error(f"语音转写失败 {video_id}: {e}")
            content_extract.audio_text = None
    
    # if req.extract_ocr:
    #     try:
    #         ocr = OCRService()
    #         content_extract.ocr_text = ocr.ocr_video(video.video_path)
    #     except Exception as e:
    #         logger.error(f"OCR识别失败 {video_id}: {e}")
    #         content_extract.ocr_text = None
    
    video_content = VideoContent(
        video_id=video_id,
        audio_text=content_extract.audio_text,
        ocr_text=content_extract.ocr_text,
        subtitle_text=content_extract.subtitle_text,
        extracted_at=datetime.now()
    )
    db.add(video_content)
    
    all_content = "\n".join([
        c for c in [content_extract.audio_text, content_extract.ocr_text, content_extract.subtitle_text] if c
    ])
    
    llm = LLMService()
    analysis = llm.analyze_content(all_content)
    
    analysis_model = AnalysisResultModel(
        video_id=video_id,
        summary=analysis.get('summary', ''),
        keywords=analysis.get('keywords', []),
        entities=analysis.get('entities', []),
        sentiment=analysis.get('sentiment', 0.0),
        categories=analysis.get('categories', []),
        analyzed_at=datetime.now()
    )
    db.add(analysis_model)
    db.commit()
    
    video.status = 'analyzed'
    db.commit()
    
    logger.info(f"视频分析完成: {video_id}")
    
    return AnalysisResult(
        video_id=video_id,
        content_extract=content_extract,
        summary=analysis.get('summary', ''),
        keywords=analysis.get('keywords', []),
        entities=analysis.get('entities', []),
        sentiment=analysis.get('sentiment', 0.0),
        categories=analysis.get('categories', [])
    )

@router.get("/summary/{video_id}", response_model=dict)
async def get_analysis_summary(video_id: str, db: Session = Depends(get_db)):
    analysis = db.query(AnalysisResultModel).filter(AnalysisResultModel.video_id == video_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="分析结果不存在")
    return {
        "video_id": analysis.video_id,
        "summary": analysis.summary,
        "keywords": analysis.keywords,
        "sentiment": analysis.sentiment,
        "categories": analysis.categories,
        "analyzed_at": analysis.analyzed_at
    }