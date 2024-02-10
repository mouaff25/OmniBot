from functools import lru_cache

from pydantic import RedisDsn, __version__
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    LOG_LEVEL: str = "INFO"
    REDIS_URL: RedisDsn = "redis://localhost:6379/0" # type: ignore
    GOOGLE_API_KEY: str
    TAVILY_API_KEY: str


@lru_cache()
def get_settings() -> Settings:
    return Settings(_env_file=".env", _env_file_encoding="utf-8") # type: ignore