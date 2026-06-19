from pydantic import BaseModel, HttpUrl
from typing import Optional, List
from datetime import datetime

class CrawlRequest(BaseModel):
    urls: List[HttpUrl]
    auto_monitor: bool = False
    monitor_interval: int = 3600

class AnalysisRequest(BaseModel):
    video_id: str
    extract_audio: bool = True
    extract_ocr: bool = True
    extract_subtitle: bool = True

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    similarity_threshold: float = 0.7

class PDFGenerateRequest(BaseModel):
    video_ids: List[str]
    template: str = "default"

class VideoUpdateRequest(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    status: Optional[str] = None