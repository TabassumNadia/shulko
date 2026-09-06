"""Central configuration. Every secret comes from .env, never from source."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM providers ---
    openai_api_key: str = ""
    google_api_key: str = ""
    tavily_api_key: str = ""

    # --- LangSmith tracing ---
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str = ""
    langchain_project: str = "shulko"

    # --- storage ---
    database_url: str = "sqlite:///./shulko.db"
    chroma_dir: str = "./data/chroma"

    # --- behaviour ---
    default_language: str = "en"           # "en" or "bn"
    confidence_threshold: float = 0.65     # below this -> needs_review
    max_line_items: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
