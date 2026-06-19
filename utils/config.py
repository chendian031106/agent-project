from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
    # PostgreSQL 配置
class Settings(BaseSettings):
    POSTGRES_HOST: str 
    POSTGRES_PORT: int 
    POSTGRES_USER: str 
    POSTGRES_PASSWORD: str 
    POSTGRES_DB: str 
    
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    
    # DashScope / 阿里云百炼
    DASHSCOPE_API_KEY: str = ""
    QWEN_API_KEY: str = ""
    QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    DOUYIN_COOKIE: str = ""
    
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_SECRET_KEY: str = "your_secret_key"
    
    VIDEO_STORAGE_PATH: Path = Path("./data/videos")
    PDF_STORAGE_PATH: Path = Path("./data/pdfs")
    MODEL_CACHE_PATH: Path = Path("./data/models")
    
    CRAWL_INTERVAL: int = 3600
    MAX_RETRY: int = 3
    TIMEOUT: int = 30
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

settings = Settings()

# 如果设置了 QWEN_API_KEY 但 DASHSCOPE_API_KEY 为空，自动填充
if not settings.DASHSCOPE_API_KEY and settings.QWEN_API_KEY:
    settings.DASHSCOPE_API_KEY = settings.QWEN_API_KEY


def get_chat_model():
    """获取聊天模型实例（优先 DeepSeek，回退 DashScope ChatOpenAI）"""
    from langchain_openai import ChatOpenAI

    if settings.DEEPSEEK_API_KEY:
        return ChatOpenAI(
            model="deepseek-chat",
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_API_BASE,
            temperature=0,
        )
    if settings.DASHSCOPE_API_KEY:
        return ChatOpenAI(
            model="qwen-plus",
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.QWEN_BASE_URL,
            temperature=0,
        )
    raise ValueError(
        "未配置任何 LLM API Key，请在 .env 中设置 DEEPSEEK_API_KEY 或 QWEN_API_KEY"
    )


def get_lightweight_model():
    """获取轻量级聊天模型（qwen-turbo，成本低、速度快）

    用于爬取、提取、RAG 等不需要强推理能力的智能体。
    """
    from langchain_openai import ChatOpenAI

    if settings.DASHSCOPE_API_KEY:
        return ChatOpenAI(
            model="qwen-turbo",
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.QWEN_BASE_URL,
            temperature=0,
        )
    # 没有 DashScope Key 时回退到标准模型
    return get_chat_model()


def get_postgres_url() -> str:
    return f"postgresql+pg8000://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"

def get_redis_url() -> str:
    if settings.REDIS_PASSWORD:
        return f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}"
    return f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"