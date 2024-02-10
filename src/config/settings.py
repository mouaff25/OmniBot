from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    LOG_LEVEL: str = "INFO"
    GOOGLE_API_KEY: str
    TAVILY_API_KEY: str


@lru_cache()
def get_settings() -> Settings:
    return Settings(_env_file=".env", _env_file_encoding="utf-8")  # type: ignore
