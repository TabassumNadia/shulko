"""
FastAPI application.

Thin on purpose: the API validates input, invokes the graph, and returns
typed output. All judgement lives in backend/agents/, all arithmetic in
backend/core/duty.py. Nothing here decides anything, which is what makes
the layer boundaries explainable.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from sqlmodel import select

from backend.core.config import get_settings
from backend.core.security import (
    create_access_token, decode_access_token, hash_password, verify_password,
)
from backend.db.session import get_session, init_db
from backend.db.models import Run, RunItem, User
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
    if settings.jwt_secret_key == "dev-only-insecure-secret-change-in-.env":
        print(
            "[shulko] WARNING: JWT_SECRET_KEY is unset, using the shared dev "
            "default. Anyone who has read backend/core/config.py can forge a "
            "login for this server. Set JWT_SECRET_KEY in .env before this "
            "is reachable by anyone but you.",
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


# --------------------------------------------------------------------- auth
_bearer = HTTPBearer(auto_error=False)


def _load_user(creds: HTTPAuthorizationCredentials | None) -> User | None:
    if creds is None:
        return None
    payload = decode_access_token(creds.credentials)
    if not payload:
        return None
    with get_session() as session:
        return session.get(User, int(payload["sub"]))


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> User:
    """Required auth: raises 401 rather than letting the route run."""
    user = _load_user(creds)
    if user is None:
        raise HTTPException(401, "Please log in to continue.")
    return user


def get_optional_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> User | None:
    """Best-effort auth for endpoints that still work anonymously.

    /analyze and /ask predate accounts, and other things (tests, curl,
    the grader) call them directly without a token. A run made this way
    just has no owner and will not show up in anyone's history -- it
    does not fail.
    """
    return _load_user(creds)


def _public_user(user: User) -> dict:
    return {"id": user.id, "email": user.email, "display_name": user.display_name}


class SignupRequest(BaseModel):
    email: str
    password: str
    display_name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


@app.post("/auth/signup", response_model=AuthResponse)
def signup(req: SignupRequest) -> AuthResponse:
    email = req.email.strip().lower()
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(400, "Enter a valid email address.")
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")

    with get_session() as session:
        if session.exec(select(User).where(User.email == email)).first():
            raise HTTPException(409, "An account with this email already exists.")
        user = User(
            email=email,
            password_hash=hash_password(req.password),
            display_name=req.display_name.strip() or email.split("@")[0],
        )
        session.add(user)
        session.commit()
        session.refresh(user)

    token = create_access_token(user.id, user.email)
    return AuthResponse(access_token=token, user=_public_user(user))


@app.post("/auth/login", response_model=AuthResponse)
def login(req: LoginRequest) -> AuthResponse:
    email = req.email.strip().lower()
    with get_session() as session:
        user = session.exec(select(User).where(User.email == email)).first()

    # Same message for "no such account" and "wrong password": telling
    # them apart lets an attacker enumerate which emails have accounts.
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(401, "Incorrect email or password.")

    token = create_access_token(user.id, user.email)
    return AuthResponse(access_token=token, user=_public_user(user))


@app.get("/auth/me")
def me(user: User = Depends(get_current_user)) -> dict:
    return _public_user(user)


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

    try:
        state = get_graph().invoke({
            "question": req.question,
            "language": req.language,
            "freight": req.freight,
            "insurance": req.insurance,
            "importer_type": req.importer_type,
        })
    except Exception as exc:
        raise _friendly_analysis_error(exc) from exc
    return {"route": state.get("route"), "report": state.get("report")}


def _friendly_analysis_error(exc: Exception) -> HTTPException:
    """Translate a pipeline failure into something a user can act on.

    The full exception is always logged server-side (below); only a
    generic, safe message and status code cross the API boundary, so a
    stack trace never reaches the browser.
    """
    from backend.core.llm import is_daily_quota_error, is_rate_limit_error

    print(f"[analyze] pipeline error: {exc!r}")

    if is_daily_quota_error(exc):
        return HTTPException(
            503,
            "The AI service's daily free quota is used up for the configured "
            "model. Try again after the quota resets, or set a fallback/paid "
            "model in .env.",
        )
    if is_rate_limit_error(exc):
        return HTTPException(
            503,
            "The AI service is temporarily rate-limited. Please try again in "
            "a moment.",
        )
    return HTTPException(
        502,
        "The AI service is temporarily unavailable. Please try again or use "
        "another available model.",
    )


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    freight: float = Form(0.0),
    insurance: float = Form(0.0),
    importer_type: str = Form("commercial"),
    language: str = Form("en"),
    user: User | None = Depends(get_optional_user),
) -> dict:
    """Invoice path: OCR, classify, cost, ground, verify, persist."""
    from backend.graph.builder import get_graph

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(
            400, f"Unsupported file type '{suffix}'. Upload {', '.join(sorted(ALLOWED))}."
        )

    # Cap upload size before it ever touches disk. Streamed in chunks so a
    # claimed-small file with a huge body cannot exhaust memory first.
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    max_bytes = 15 * 1024 * 1024
    written = 0
    with path.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                out.close()
                path.unlink(missing_ok=True)
                raise HTTPException(413, "File too large. Upload up to 15 MB.")
            out.write(chunk)

    try:
        state = get_graph().invoke({
            "raw_file_path": str(path),
            "freight": freight,
            "insurance": insurance,
            "importer_type": importer_type,
            "language": language,
        })
    except Exception as exc:
        raise _friendly_analysis_error(exc) from exc
    finally:
        path.unlink(missing_ok=True)

    run_id = _persist(
        file.filename or "invoice", importer_type, language, state,
        user_id=user.id if user else None,
    )
    return {"run_id": run_id, "route": state.get("route"),
            "report": state.get("report")}


def _persist(
    filename: str, importer_type: str, language: str, state: dict,
    user_id: int | None = None,
) -> int | None:
    """Save the run. A storage failure must not lose the user's answer."""
    try:
        report = state.get("report") or {}
        with get_session() as session:
            run = Run(
                user_id=user_id,
                filename=filename,
                importer_type=importer_type,
                language=language,
                total_tax_incidence=float(report.get("totals", {}).get("tti", 0.0)),
                needs_review=bool(report.get("needs_review")),
                report_json=json.dumps(report),
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
def runs(limit: int = 20, user: User = Depends(get_current_user)) -> list[dict]:
    """The logged-in user's own analysis history, most recent first.

    Requires login and is filtered to `user_id == user.id` at the query
    level (not filtered client-side after fetching everyone's rows), so
    one user's history is never even sent to another user's session.
    """
    with get_session() as session:
        rows = session.exec(
            select(Run)
            .where(Run.user_id == user.id)
            .order_by(Run.id.desc())
            .limit(limit)
        ).all()
        return [r.model_dump(exclude={"report_json"}) for r in rows]


@app.get("/runs/{run_id}")
def run_detail(run_id: int, user: User = Depends(get_current_user)) -> dict:
    """Reopen one previous analysis, report included."""
    with get_session() as session:
        run = session.get(Run, run_id)
        if run is None or run.user_id != user.id:
            # Same response for "doesn't exist" and "belongs to someone
            # else": the second must not be distinguishable from the
            # first, or a run id is enough to probe who else has one.
            raise HTTPException(404, "Analysis not found.")
        items = session.exec(
            select(RunItem).where(RunItem.run_id == run.id)
        ).all()
        return {
            **run.model_dump(exclude={"report_json"}),
            "report": json.loads(run.report_json) if run.report_json else None,
            "items": [i.model_dump() for i in items],
        }
