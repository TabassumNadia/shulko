"""
Shulko backend.

Importing this package configures LangSmith tracing, and that eager call
is the whole reason this file has contents.

LangChain and LangSmith both decide whether to trace by reading
os.environ at the moment a run starts. Our keys live in .env, and
`get_settings()` is what copies them into the environment — so tracing
was silently on or off depending on whether some *other* module happened
to call `get_settings()` first.

That is exactly the bug it looks like: running the graph directly
produced no traces at all, while running it behind the API produced
full ones, because `backend.db.session` calls `get_settings()` at import
and the FastAPI app imports it. Same code, same key, different entry
point, and nothing said anything was wrong.

Doing it here means any entry point — the API, the Streamlit front-end,
a test, `python -m backend.rag.ingest` — has tracing configured before
it can start a run. `get_settings()` is `lru_cache`d, so this costs one
.env read for the process and nothing thereafter.
"""

from backend.core.config import get_settings

# Called for the side effect: configure_tracing() runs inside and
# publishes LANGSMITH_* into os.environ where the tracers read them.
get_settings()
