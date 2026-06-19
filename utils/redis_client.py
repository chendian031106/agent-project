"""
Redis 客户端

使用单例模式，延迟连接（连接失败时不会抛出异常）。
所有操作在连接不可用时安全降级，返回 None / 空值。
"""

import redis
from utils.config import settings, get_redis_url
from utils.logger import logger


class RedisClient:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisClient, cls).__new__(cls)
            cls._instance._client = None
            cls._instance._connected = False
        return cls._instance

    def _ensure_connected(self):
        """延迟连接：首次使用时才尝试连接 Redis"""
        if self._connected:
            return True
        if self._client is not None:
            # 已经尝试过但失败了，不再重试
            return False
        return self._connect()

    def _connect(self):
        """尝试建立 Redis 连接"""
        try:
            self._client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self._client.ping()
            self._connected = True
            logger.info("Redis 连接成功")
            return True
        except Exception as e:
            logger.warning(f"Redis 不可用（功能将降级）: {e}")
            self._client = None
            self._connected = False
            return False

    # ---------- 通用操作 ----------

    def get(self, key: str):
        if not self._ensure_connected():
            return None
        try:
            return self._client.get(key)
        except Exception as e:
            logger.warning(f"Redis get 失败: {e}")
            return None

    def set(self, key: str, value: str, ex: int = None):
        if not self._ensure_connected():
            return None
        try:
            return self._client.set(key, value, ex=ex)
        except Exception as e:
            logger.warning(f"Redis set 失败: {e}")
            return None

    def delete(self, key: str):
        if not self._ensure_connected():
            return None
        try:
            return self._client.delete(key)
        except Exception as e:
            logger.warning(f"Redis delete 失败: {e}")
            return None

    def exists(self, key: str) -> bool:
        if not self._ensure_connected():
            return False
        try:
            return self._client.exists(key) > 0
        except Exception as e:
            logger.warning(f"Redis exists 失败: {e}")
            return False

    # ---------- Hash 操作 ----------

    def hget(self, name: str, key: str):
        if not self._ensure_connected():
            return None
        try:
            return self._client.hget(name, key)
        except Exception as e:
            logger.warning(f"Redis hget 失败: {e}")
            return None

    def hset(self, name: str, key: str, value: str):
        if not self._ensure_connected():
            return None
        try:
            return self._client.hset(name, key, value)
        except Exception as e:
            logger.warning(f"Redis hset 失败: {e}")
            return None

    # ---------- List 操作 ----------

    def lpush(self, name: str, *values):
        if not self._ensure_connected():
            return None
        try:
            return self._client.lpush(name, *values)
        except Exception as e:
            logger.warning(f"Redis lpush 失败: {e}")
            return None

    def rpop(self, name: str):
        if not self._ensure_connected():
            return None
        try:
            return self._client.rpop(name)
        except Exception as e:
            logger.warning(f"Redis rpop 失败: {e}")
            return None

    # ---------- 其他 ----------

    def incr(self, key: str) -> int:
        if not self._ensure_connected():
            return 0
        try:
            return self._client.incr(key)
        except Exception as e:
            logger.warning(f"Redis incr 失败: {e}")
            return 0

    def expire(self, key: str, time: int):
        if not self._ensure_connected():
            return None
        try:
            return self._client.expire(key, time)
        except Exception as e:
            logger.warning(f"Redis expire 失败: {e}")
            return None


# 模块级单例（不会再导入时连接）
redis_client = RedisClient()