from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from utils.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != settings.API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的API密钥"
        )
    return api_key