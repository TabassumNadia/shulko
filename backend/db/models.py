"""SQLite persistence — analysis history."""

from datetime import datetime
from sqlmodel import SQLModel, Field


class Run(SQLModel, table=True):
    """One invoice analysis."""
    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    filename: str
    importer_type: str = "commercial"
    language: str = "en"
    total_tax_incidence: float = 0.0
    needs_review: bool = False


class RunItem(SQLModel, table=True):
    """One line item inside a run."""
    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id")
    description: str
    hs_code: str | None = None
    confidence: float = 0.0
    tti: float = 0.0
