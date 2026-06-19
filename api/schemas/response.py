from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict
from datetime import datetime

class VideoInfo(BaseModel):
    id: str
    title: str
    url: str
    cover_url: Optional[str]
    duration: int
    author: str
    view_count: int
    like_count: int
    comment_count: int
    video_path: Optional[str]
    crawl_time: datetime
    status: str

class ContentExtract(BaseModel):
    audio_text: Optional[str]
    ocr_text: Optional[str]
    subtitle_text: Optional[str]

class AnalysisResult(BaseModel):
    video_id: str
    content_extract: ContentExtract
    summary: str
    keywords: List[str]
    entities: List[Dict[str, str]]
    sentiment: float
    categories: List[str]

class QueryResult(BaseModel):
    answer: str
    sources: List[Dict[str, str]]
    confidence: float

class PDFResponse(BaseModel):
    pdf_id: str
    download_url: str
    generated_at: datetime

class CrawlTaskInfo(BaseModel):
    id: int
    url: str
    status: str
    last_run_at: Optional[datetime]
    next_run_at: Optional[datetime]
    created_at: datetime

class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None

class VideoListResponse(BaseModel):
    items: List[VideoInfo]
    total: int
    page: int
    page_size: int