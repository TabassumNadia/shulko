"""FastAPI entry point. Wired up in the H+9 block."""

from fastapi import FastAPI

from backend.db.session import init_db

app = FastAPI(title="Shulko API", version="0.1.0")


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


# TODO H+9  POST /analyze  -> run the invoice graph
# TODO H+13 POST /ask      -> run the question graph
