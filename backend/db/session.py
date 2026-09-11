"""Database engine and session helpers."""

from sqlmodel import SQLModel, Session, create_engine, text

from backend.core.config import get_settings

_settings = get_settings()
engine = create_engine(_settings.database_url, echo=False)


def init_db() -> None:
    """Create tables if they do not exist, then patch older ones.

    `create_all` only adds tables that are missing entirely -- an
    existing `run` table from before accounts/history existed keeps its
    original columns forever. The migration below is intentionally tiny
    (additive, SQLite-only, `ALTER TABLE ... ADD COLUMN`) rather than a
    full migration tool, because a demo project has one schema change to
    make, not a history of them.
    """
    SQLModel.metadata.create_all(engine)
    _add_missing_columns()


def _add_missing_columns() -> None:
    additions = {
        "run": {
            "user_id": "INTEGER",
            "report_json": "TEXT DEFAULT ''",
        },
    }
    with engine.connect() as conn:
        for table, columns in additions.items():
            existing = {
                row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))
            }
            for column, ddl_type in columns.items():
                if column not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))
        conn.commit()


def get_session() -> Session:
    return Session(engine)
