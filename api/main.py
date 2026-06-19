from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import crawler, videos, analysis, knowledge, pdf
from database.session import init_db
from utils.logger import logger

app = FastAPI(title="抖音多智能体内容聚合系统", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(crawler.router, prefix="/api")
app.include_router(videos.router, prefix="/api")
app.include_router(analysis.router, prefix="/api")
app.include_router(knowledge.router, prefix="/api")
app.include_router(pdf.router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    try:
        init_db()
        logger.info("数据库初始化完成")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")

@app.get("/")
async def root():
    return {"message": "抖音多智能体内容聚合系统 API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}