"""
Central configuration. Every secret comes from .env, never from source.

LangSmith accepts two generations of variable names. The current SDK
documents LANGSMITH_*; LANGCHAIN_* is the legacy spelling that still
works. Both are read here and the new one wins, so a .env copied from
either era of the docs behaves the same. The setter below also pushes
them into the process environment, because the tracing client reads
os.environ directly rather than this object.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM providers ---
    google_api_key: str = ""
    tavily_api_key: str = ""

    # --- LangSmith tracing: current names ---
    langsmith_api_key: str = ""
    langsmith_project: str = ""
    langsmith_tracing: bool = False
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # --- LangSmith tracing: legacy names, still honoured ---
    langchain_api_key: str = ""
    langchain_project: str = ""
    langchain_tracing_v2: bool = False
    langchain_endpoint: str = ""

    # --- models ---
    # Google retires model ids on a schedule (text-embedding-004 and
    # gemini-2.0-flash both went during this project). Every id is
    # configuration, so a shutdown is a .env edit, not a code change.
    # Check what your key can reach: python -m backend.core.llm
    chat_model: str = "gemini-3.5-flash"          # classification
    vision_model: str = "gemini-3.5-flash"        # invoice OCR
    judge_model: str = "gemini-3.5-flash-lite"    # verification, routing
    grounding_model: str = "gemini-3.5-flash"     # google_search tool

    # --- embeddings ---
    # Local by default. The Gemini free tier allows 100 embedding requests
    # a minute; embedding 7,154 tariff rows through it burns the same quota
    # the agents need for classification and grounding. Bulk-embedding a
    # static price list needs no intelligence, so it should not compete
    # for the intelligent budget. Set to "google" to switch.
    embedding_provider: str = "fastembed"          # fastembed | google
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    embedding_model: str = "models/gemini-embedding-001"   # google path only

    # --- storage ---
    database_url: str = "sqlite:///./shulko.db"
    chroma_dir: str = "./data/chroma"

    # --- behaviour ---
    default_language: str = "en"           # "en" or "bn"
    confidence_threshold: float = 0.65     # below this -> needs_review
    max_line_items: int = 10
    # Grounded search is the slowest call in the pipeline. Cap it per run
    # so a long invoice cannot turn one analysis into a five-minute wait.
    max_regulatory_checks: int = 2
    # Seconds before a single model call is abandoned. Long enough for a
    # vision call on a full-page invoice, short enough that a stalled
    # request surfaces as an error instead of an endless spinner.
    llm_timeout: int = 90

    # --- quota ---
    # The Gemini free tier allows 5 generateContent requests a minute, per
    # model. One invoice needs far more than five calls, and they run
    # concurrently, so without a shared throttle the run dies on a 429
    # partway through -- the single most common way this pipeline fails.
    #
    # Every model call in the process passes through one token bucket
    # sized from this number. Raise it if your key is on a paid tier;
    # the pipeline gets faster with no code change.
    gemini_rpm: int = 5
    # How many line items are worked on at once. Kept at or below the
    # per-minute budget: more threads than the quota allows only means
    # more requests queueing behind the same bucket.
    max_workers: int = 3
    # A 429 that slips through the bucket is retried rather than fatal.
    llm_max_attempts: int = 4

    # ------------------------------------------------------- resolved view
    @property
    def tracing_api_key(self) -> str:
        return self.langsmith_api_key or self.langchain_api_key

    @property
    def tracing_project(self) -> str:
        return self.langsmith_project or self.langchain_project or "shulko"

    @property
    def tracing_endpoint(self) -> str:
        return self.langsmith_endpoint or self.langchain_endpoint

    @property
    def tracing_enabled(self) -> bool:
        """On when a key exists. The tracing flags can only turn it OFF.

        Defaulting to on-with-a-key avoids the commonest setup mistake:
        a valid key in .env, no explicit flag, and no traces to submit.
        """
        if not self.tracing_api_key:
            return False
        explicit = self.langsmith_tracing or self.langchain_tracing_v2
        return explicit or not self._tracing_explicitly_disabled()

    def _tracing_explicitly_disabled(self) -> bool:
        import os
        raw = (os.getenv("LANGSMITH_TRACING")
               or os.getenv("LANGCHAIN_TRACING_V2") or "").strip().lower()
        return raw in {"false", "0", "no"}


def configure_tracing(settings: Settings) -> bool:
    """Publish tracing settings into os.environ.

    The LangSmith client reads the environment, not our Settings object,
    so without this a key sitting in .env would be silently ignored and
    no trace would ever appear. Returns whether tracing is actually on,
    so callers can say so out loud instead of guessing.
    """
    if not settings.tracing_enabled:
        # Make the off state explicit rather than leaving a stale value
        # from the shell to decide.
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    for new, legacy, value in (
        ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", settings.tracing_api_key),
        ("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT", settings.tracing_project),
        ("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT", settings.tracing_endpoint),
    ):
        if value:
            os.environ[new] = value
            os.environ[legacy] = value

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    return True


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    configure_tracing(settings)
    return settings
