"""
FastAPI application.

Thin on purpose: the API validates input, invokes the graph, and returns
typed output. All judgement lives in backend/agents/, all arithmetic in
backend/core/duty.py. Nothing here decides anything, which is what makes
the layer boundaries explainable.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from backend.core.config import get_settings
from backend.db.session import get_session, init_db
from backend.db.models import Run, RunItem
from backend.tools.tariff_tool import loaded_chapters

UPLOAD_DIR = Path(tempfile.gettempdir()) / "shulko-uploads"
ALLOWED = {".png", ".jpg", ".jpeg", ".webp", ".pdf"}


def _log_effective_config() -> None:
    """Print the models this process will actually use.

    Settings are read once, at import, and `uvicorn --reload` watches
    *.py only -- editing .env does not restart the server. So a process
    can outlive the configuration that started it, and /health will keep
    faithfully reporting the models this process loaded while .env on disk
    says something else. Printing them at startup makes the effective
    config checkable at a glance, and makes the fix obvious: restart.
    """
    settings = get_settings()
    print(
        "[shulko] models in effect -- "
        f"chat={settings.chat_model} "
        f"vision={settings.vision_model} "
        f"judge={settings.judge_model} "
        f"grounding={settings.grounding_model} "
        f"embeddings={settings.embedding_provider}",
        flush=True,
    )
    print(
        f"[shulko] tracing={'on' if settings.tracing_enabled else 'off'} "
        f"project={settings.tracing_project} -- "
        "edit .env then RESTART for changes to apply "
        "(uvicorn --reload does not watch .env)",
        flush=True,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Startup work. `on_event` is deprecated in current FastAPI, so the
    database and upload directory are prepared here instead."""
    init_db()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _log_effective_config()
    yield


app = FastAPI(
    title="Shulko API",
    version="0.1.0",
    description="Bangladesh import landed-cost agent.",
    lifespan=lifespan,
)


def _index_sizes() -> dict:
    """How many documents are really in each collection.

    An empty vector store degrades retrieval to BM25 silently, which is
    exactly the kind of thing you want to notice before recording a demo,
    not during it.
    """
    try:
        from backend.rag.store import (
            NOTES_COLLECTION, TARIFF_COLLECTION, collection_size,
        )
        return {
            "tariff_lines": collection_size(TARIFF_COLLECTION),
            "chapter_notes": collection_size(NOTES_COLLECTION),
        }
    except Exception:
        return {"tariff_lines": 0, "chapter_notes": 0}


def _notes_chapters() -> list[str]:
    """Chapters with Section/Chapter Notes loaded, from the live cache."""
    try:
        from backend.agents.classifier import chapters_with_notes
        return chapters_with_notes()
    except Exception:
        return []


@app.get("/health")
def health() -> dict:
    """Startup check. Say plainly what is and is not configured, so a
    missing key surfaces here rather than as a confusing failure later."""
    settings = get_settings()
    try:
        chapters = loaded_chapters()
    except Exception:
        chapters = []
    try:
        from backend.rag.retriever import get_retriever
        retrieval_mode = get_retriever().mode
    except Exception:
        retrieval_mode = "unavailable"

    return {
        "status": "ok",
        "tariff_chapters_loaded": len(chapters),
        "gemini_key_set": bool(settings.google_api_key),
        "indexed": _index_sizes(),
        "models": {
            "embedding_provider": settings.embedding_provider,
            "chat": settings.chat_model,
            "vision": settings.vision_model,
            "judge": settings.judge_model,
            "embedding": settings.embedding_model,
        },
        # Which chapters the Chapter-Note knowledge base actually covers.
        # An item outside this list can never carry a note citation, so the
        # limit belongs in the open rather than looking like a retrieval bug.
        "notes_chapters": _notes_chapters(),
        "retrieval_mode": retrieval_mode,
        "langsmith_tracing": settings.tracing_enabled,
        "langsmith_project": settings.tracing_project,
    }


class AskRequest(BaseModel):
    question: str
    language: str = "en"
    freight: float = 0.0
    insurance: float = 0.0
    importer_type: str = "commercial"


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    """Question path: route, classify, cost. No file involved."""
    from backend.graph.builder import get_graph

    state = get_graph().invoke({
        "question": req.question,
        "language": req.language,
        "freight": req.freight,
        "insurance": req.insurance,
        "importer_type": req.importer_type,
    })
    return {"route": state.get("route"), "report": state.get("report")}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    freight: float = Form(0.0),
    insurance: float = Form(0.0),
    importer_type: str = Form("commercial"),
    language: str = Form("en"),
) -> dict:
    """Invoice path: OCR, classify, cost, ground, verify, persist."""
    from backend.graph.builder import get_graph

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(
            400, f"Unsupported file type '{suffix}'. Upload {', '.join(sorted(ALLOWED))}."
        )

    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        state = get_graph().invoke({
            "raw_file_path": str(path),
            "freight": freight,
            "insurance": insurance,
            "importer_type": importer_type,
            "language": language,
        })
    except Exception as exc:
        raise HTTPException(500, f"Analysis failed: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)

    run_id = _persist(file.filename or "invoice", importer_type, language, state)
    return {"run_id": run_id, "route": state.get("route"),
            "report": state.get("report")}


def _persist(filename: str, importer_type: str, language: str, state: dict) -> int | None:
    """Save the run. A storage failure must not lose the user's answer."""
    try:
        report = state.get("report") or {}
        with get_session() as session:
            run = Run(
                filename=filename,
                importer_type=importer_type,
                language=language,
                total_tax_incidence=float(report.get("totals", {}).get("tti", 0.0)),
                needs_review=bool(report.get("needs_review")),
            )
            session.add(run)
            session.commit()
            session.refresh(run)

            for r in state.get("results", []):
                top = (r.get("candidates") or [{}])[0]
                session.add(RunItem(
                    run_id=run.id,
                    description=r["item"]["description"][:300],
                    hs_code=r.get("chosen_hs_code"),
                    confidence=float(top.get("confidence", 0.0)),
                    tti=float((r.get("duty") or {}).get("tti", 0.0)),
                ))
            session.commit()
            return run.id
    except Exception as exc:
        print(f"[persist] could not save run: {exc}")
        return None


@app.get("/runs")
def runs(limit: int = 20) -> list[dict]:
    """Analysis history, straight from SQLite."""
    from sqlmodel import select

    with get_session() as session:
        rows = session.exec(
            select(Run).order_by(Run.id.desc()).limit(limit)
        ).all()
        return [r.model_dump() for r in rows]
