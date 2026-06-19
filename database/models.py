from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Video(Base):
    __tablename__ = 'videos'
    
    id = Column(String, primary_key=True)
    title = Column(String(500))
    url = Column(String(500))
    cover_url = Column(String(500))
    duration = Column(Integer)
    author = Column(String(200))
    view_count = Column(Integer)
    like_count = Column(Integer)
    comment_count = Column(Integer)
    video_path = Column(String(500))
    crawl_time = Column(DateTime, default=datetime.now)
    status = Column(String(50), default='pending')
    
    content = relationship("VideoContent", back_populates="video", uselist=False)
    analysis = relationship("AnalysisResult", back_populates="video", uselist=False)
    knowledge_entries = relationship("KnowledgeBase", back_populates="video")

class VideoContent(Base):
    __tablename__ = 'video_content'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String, ForeignKey('videos.id'))
    audio_text = Column(Text)
    ocr_text = Column(Text)
    subtitle_text = Column(Text)
    extracted_at = Column(DateTime, default=datetime.now)
    
    video = relationship("Video", back_populates="content")

class AnalysisResult(Base):
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String, ForeignKey('videos.id'))
    summary = Column(Text)
    keywords = Column(JSON)
    entities = Column(JSON)
    sentiment = Column(Float)
    categories = Column(JSON)
    analyzed_at = Column(DateTime, default=datetime.now)
    
    video = relationship("Video", back_populates="analysis")

class KnowledgeBase(Base):
    __tablename__ = 'knowledge_base'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String, ForeignKey('videos.id'), nullable=True)
    content = Column(Text, nullable=False)
    embedding = Column(Text, nullable=True)  # 存储 JSON 数组（向量），在 Python 层做余弦相似度
    title = Column(String(500), default="")
    author = Column(String(200), default="")
    tags = Column(JSON, default=list)
    source_url = Column(String(500), default="")
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    video = relationship("Video", back_populates="knowledge_entries")

class CrawlTask(Base):
    __tablename__ = 'crawl_tasks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String(500))
    status = Column(String(50), default='pending')
    last_run_at = Column(DateTime)
    next_run_at = Column(DateTime)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.now)

class PDFGeneration(Base):
    __tablename__ = 'pdf_generations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_ids = Column(JSON)
    pdf_path = Column(String(500))
    generated_at = Column(DateTime, default=datetime.now)