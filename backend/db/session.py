"""Database engine and session helpers."""

from sqlmodel import SQLModel, Session, create_engine

from backend.core.config import get_settings

_settings = get_settings()
engine = create_engine(_settings.database_url, echo=False)


def init_db() -> None:
    """Create tables if they do not exist. Call once at startup."""
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    return Session(engine)
