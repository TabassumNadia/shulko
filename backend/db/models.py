"""SQLite persistence — users and analysis history."""

from datetime import datetime
from sqlmodel import SQLModel, Field


class User(SQLModel, table=True):
    """A Shulko account. Never holds a plaintext password."""
    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    email: str = Field(index=True, unique=True)
    password_hash: str
    display_name: str = ""


class Run(SQLModel, table=True):
    """One invoice analysis (or duty question)."""
    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    # Nullable so old rows created before accounts existed, and any run
    # made through the API without a token, still have a home. `/runs`
    # only ever returns rows that match the caller's own id.
    user_id: int | None = Field(default=None, foreign_key="user.id", index=True)
    filename: str
    importer_type: str = "commercial"
    language: str = "en"
    total_tax_incidence: float = 0.0
    needs_review: bool = False
    # The full report the user originally saw (JSON text), so "reopen"
    # in the dashboard shows the real analysis -- HS candidates, evidence,
    # advisories, duty breakdown -- not just the four summary columns
    # below. RunItem stays as the normalized per-line summary used for
    # quick history rows and the /runs list.
    report_json: str = ""


class RunItem(SQLModel, table=True):
    """One line item inside a run."""
    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id")
    description: str
    hs_code: str | None = None
    confidence: float = 0.0
    tti: float = 0.0
