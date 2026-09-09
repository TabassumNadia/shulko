"""
Vector store — LangChain Chroma, two collections.

Chunking, and why it is what it is:

    tariff_lines    one HS row = one chunk
        A row is already the smallest meaningful unit. Splitting one
        would separate a rate from the description it belongs to, and
        the rates live in metadata so they survive retrieval intact.

    chapter_notes   one note = one chunk
        Notes are self-contained legal sentences the classifier must
        quote verbatim. Cutting one mid-sentence would destroy the
        quote, so they are never split regardless of length.

Embeddings run LOCALLY by default, and that is a deliberate call rather
than a cost saving. The Gemini free tier allows 100 embedding requests a
minute; embedding 7,154 tariff rows through it exhausts the same quota
the agents need for classification and Google Search grounding. Spending
an API budget on turning a static price list into vectors — work that
never changes and needs no intelligence — starves the part of the system
that actually reasons.

So: FastEmbed (ONNX, ~90 MB, no torch, no network after first run) does
the bulk embedding, and every Gemini call is left for the agents. Set
EMBEDDING_PROVIDER=google in .env to switch back.

Everything goes through LangChain's VectorStore interface, so retrieval
appears in LangSmith traces automatically.
"""

from __future__ import annotations

import functools
import time

from langchain_core.documents import Document

from backend.core.config import get_settings

TARIFF_COLLECTION = "tariff_lines"
NOTES_COLLECTION = "chapter_notes"

EMBED_BATCH = 200          # local embedding: batch for throughput
GOOGLE_BATCH = 90          # google: stay under the 100/min request quota
GOOGLE_PAUSE = 61          # seconds between batches, quota window + margin
EMBED_DIMENSIONS = 768     # gemini-embedding-001 defaults to 3072


# --------------------------------------------------------------- embeddings
@functools.lru_cache(maxsize=4)
def get_embeddings():
    """Embedding model for the configured provider.

    Cached because building a FastEmbedEmbeddings loads a ~90 MB ONNX
    model from disk. Without this, every call that touches the store —
    and the classifier touches it once per chapter, per line item —
    reloads the model, which is seconds of pure waiting per invoice.

    Raises:
        RuntimeError: naming the package or key to fix, rather than
            surfacing a library stack trace.
    """
    settings = get_settings()
    provider = settings.embedding_provider.lower()

    if provider == "fastembed":
        try:
            from langchain_community.embeddings import FastEmbedEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "FastEmbed is not installed. Run: pip install fastembed\n"
                "Or set EMBEDDING_PROVIDER=google in .env to use the API "
                "instead (slower, and it consumes your Gemini quota)."
            ) from exc
        return FastEmbedEmbeddings(model_name=settings.fastembed_model)

    if provider == "google":
        if not settings.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to .env "
                "(free key at aistudio.google.com/apikey)."
            )
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
            output_dimensionality=EMBED_DIMENSIONS,
        )

    raise RuntimeError(
        f"Unknown EMBEDDING_PROVIDER {provider!r}. Use 'fastembed' or 'google'."
    )


def collection_name(base: str) -> str:
    """Namespace a collection by embedding provider.

    A Chroma collection fixes its vector dimensionality when it is
    created: 768 for Gemini embeddings, 384 for bge-small. Reusing one
    name across providers fails with "expecting embedding with dimension
    of 768, got 384" halfway through a bulk index.

    Putting the provider in the name means switching providers builds a
    fresh collection alongside the old one instead of colliding with it,
    and switching back finds the previous index still intact.
    """
    return f"{base}__{get_settings().embedding_provider.lower()}"


@functools.lru_cache(maxsize=8)
def get_vectorstore(base: str):
    """A persistent LangChain Chroma collection for the active provider.

    Cached for the same reason as the embeddings: opening a collection is
    not free, and it happens on every note lookup.
    """
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection_name(base),
        embedding_function=get_embeddings(),
        persist_directory=get_settings().chroma_dir,
    )


# ----------------------------------------------------------------- write
def _tariff_documents(rows: list[dict]) -> list[Document]:
    """One Document per HS row. Rates ride in metadata, not in the text.

    Keeping numbers out of the embedded text matters: '15' and '5' carry
    no semantic signal, and including them only blurs the vector.
    """
    return [
        Document(
            page_content=f"{r['hs_code']} {r['description']}",
            metadata={
                "hs_code": r["hs_code"],
                "chapter": r["chapter"],
                "heading": r["heading"],
                "description": r["description"],
                "cd": float(r.get("cd") or 0),
                "rd": float(r.get("rd") or 0),
                "sd": float(r.get("sd") or 0),
                "vat": float(r.get("vat") or 15),
                "ait": float(r.get("ait") or 5),
                "at": float(r.get("at") or 5),
            },
        )
        for r in rows
    ]


def _note_documents(notes: list[dict]) -> list[Document]:
    """One Document per note, kept whole so it can be quoted verbatim."""
    return [
        Document(
            page_content=n["text"],
            metadata={
                "chapter": n["chapter"],
                "note_no": n["note_no"],
                "note_type": n["note_type"],
                "scope": n.get("scope", "chapter"),
                "source": n["source"],
            },
        )
        for n in notes
    ]


def existing_ids(base: str) -> set[str]:
    """Ids already in the collection, so a re-run resumes.

    Bulk indexing gets interrupted — a rate limit, a closed laptop, a
    Ctrl-C. Restarting from zero each time wastes both the quota and the
    evening.
    """
    try:
        store = get_vectorstore(base)
        return set(store.get(include=[]).get("ids", []))
    except Exception:
        return set()


def _add_in_batches(store, docs: list[Document], ids: list[str],
                    batch_size: int, pause: float) -> int:
    added = 0
    for i in range(0, len(docs), batch_size):
        chunk, chunk_ids = docs[i:i + batch_size], ids[i:i + batch_size]
        try:
            store.add_documents(chunk, ids=chunk_ids)
        except Exception as exc:
            if "dimension" in str(exc).lower():
                raise RuntimeError(
                    f"{exc}\n\n"
                    "The collection was built with a different embedding "
                    "model. Delete data/chroma and index again, or switch "
                    "EMBEDDING_PROVIDER back in .env."
                ) from exc
            raise
        added += len(chunk)
        print(f"  embedded {added}/{len(docs)}", flush=True)
        if pause and i + batch_size < len(docs):
            print(f"  pausing {pause:.0f}s for the API quota window", flush=True)
            time.sleep(pause)
    return added


def index_tariff_lines(rows: list[dict], resume: bool = True) -> int:
    """Embed and store tariff rows. Returns how many were newly added."""
    settings = get_settings()
    store = get_vectorstore(TARIFF_COLLECTION)

    docs = _tariff_documents(rows)
    ids = [d.metadata["hs_code"] for d in docs]

    if resume:
        done = existing_ids(TARIFF_COLLECTION)
        if done:
            keep = [(d, i) for d, i in zip(docs, ids) if i not in done]
            print(f"  {len(done)} already indexed, {len(keep)} to go")
            docs = [d for d, _ in keep]
            ids = [i for _, i in keep]

    if not docs:
        return 0

    on_google = settings.embedding_provider.lower() == "google"
    return _add_in_batches(
        store, docs, ids,
        batch_size=GOOGLE_BATCH if on_google else EMBED_BATCH,
        pause=GOOGLE_PAUSE if on_google else 0,
    )


def _unique_ids(notes: list[dict]) -> list[str]:
    """Stable, collision-free ids for notes.

    Chapters carry two separately numbered note blocks — "Notes" and
    "Subheading Notes" — so chapter+number alone is not unique, and Chroma
    rejects the whole batch on a duplicate. Scope disambiguates them; the
    counter is a backstop so a parsing quirk degrades one id rather than
    failing the entire index.
    """
    ids, seen = [], {}
    for n in notes:
        base = f"{n['chapter']}-{n.get('scope', 'chapter')}-{n['note_no']}"
        seen[base] = seen.get(base, 0) + 1
        ids.append(base if seen[base] == 1 else f"{base}-{seen[base]}")
    return ids


def index_chapter_notes(notes: list[dict], resume: bool = True) -> int:
    settings = get_settings()
    store = get_vectorstore(NOTES_COLLECTION)

    docs = _note_documents(notes)
    ids = _unique_ids(notes)

    if resume:
        done = existing_ids(NOTES_COLLECTION)
        keep = [(d, i) for d, i in zip(docs, ids) if i not in done]
        docs = [d for d, _ in keep]
        ids = [i for _, i in keep]

    if not docs:
        return 0

    on_google = settings.embedding_provider.lower() == "google"
    return _add_in_batches(
        store, docs, ids,
        batch_size=GOOGLE_BATCH if on_google else EMBED_BATCH,
        pause=GOOGLE_PAUSE if on_google else 0,
    )


# ------------------------------------------------------------------ read
def notes_for_chapter(chapter: str) -> list[dict]:
    """Every note for a chapter — deliberately not a top-k search.

    The classifier has to see all the notes that could exclude a
    candidate. Retrieving only the most similar few would silently drop
    the exclusion that matters most.
    """
    store = get_vectorstore(NOTES_COLLECTION)
    got = store.get(where={"chapter": chapter})
    return [
        {"text": text, **meta}
        for text, meta in zip(got.get("documents", []), got.get("metadatas", []))
    ]


def collection_size(base: str) -> int:
    """How many documents are actually indexed. Surfaced in /health so an
    empty index is visible rather than silently degrading retrieval."""
    try:
        return len(get_vectorstore(base).get(include=[]).get("ids", []))
    except Exception:
        return 0
