"""
数据库会话管理 — 延迟初始化模式

避免在模块加载时就创建数据库连接，
防止 PostgreSQL 未运行时因 Windows 下 GBK 编码错误导致整个应用无法启动。
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from utils.config import get_postgres_url
from utils.logger import logger
from database.models import Base

# 延迟初始化：不在模块加载时创建 engine
_engine = None
_SessionLocal = None


def get_engine():
    """获取数据库引擎（首次调用时创建）"""
    global _engine
    if _engine is None:
        try:
            _engine = create_engine(get_postgres_url(), pool_pre_ping=True)
        except Exception as e:
            logger.warning(f"数据库引擎创建失败（将以无数据库模式运行）: {e}")
            _engine = None
    return _engine


def get_session_factory():
    """获取 Session 工厂（首次调用时创建）"""
    global _SessionLocal
    if _SessionLocal is None:
        eng = get_engine()
        if eng is not None:
            _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=eng)
    return _SessionLocal


def SessionLocal():
    """获取数据库会话（兼容旧版直接调用）"""
    factory = get_session_factory()
    if factory is None:
        raise RuntimeError("数据库不可用，请检查 PostgreSQL 是否运行")
    return factory()


def init_db():
    """初始化数据库表结构"""
    try:
        eng = get_engine()
        if eng is None:
            logger.warning("数据库引擎不可用，跳过表初始化")
            return
        # 测试连接
        with eng.connect() as conn:
            pass
        Base.metadata.create_all(bind=eng)
        logger.info("数据库表创建成功")
    except Exception as e:
        # Windows 中文系统下 PostgreSQL 未运行时可能返回 GBK 编码错误
        err_msg = str(e)
        if "codec can't decode" in err_msg or "invalid continuation byte" in err_msg:
            logger.warning("数据库连接失败（编码错误，PostgreSQL 可能未运行）")
        else:
            logger.warning(f"数据库初始化失败（将以无数据库模式运行）: {err_msg}")


def get_db():
    """获取数据库会话（FastAPI Depends 使用）"""
    factory = get_session_factory()
    if factory is None:
        raise RuntimeError("数据库不可用，请检查 PostgreSQL 是否运行")
    db = factory()
    try:
        yield db
    finally:
        db.close()


# 兼容旧代码：提供 SessionLocal 属性
class _SessionLocalProxy:
    def __call__(self):
        factory = get_session_factory()
        if factory is None:
            raise RuntimeError("数据库不可用")
        return factory()


SessionLocal = _SessionLocalProxy()