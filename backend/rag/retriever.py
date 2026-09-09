"""
Hybrid retrieval — LangChain EnsembleRetriever (BM25 + dense).

This is Stage 2 of the deterministic workflow in arXiv 2605.14857, built
on LangChain's documented hybrid-search components rather than hand-rolled
ranking:

    BM25Retriever       lexical  — exact terms and code fragments
    Chroma.as_retriever dense    — meaning, when the wording differs
    EnsembleRetriever   fuses both with Reciprocal Rank Fusion

Why both engines? Tariff descriptions mix precise technical vocabulary
with loose trade language. BM25 nails "polypropylene" and "3901"; the
dense side catches "plastic bag for packing goods" when the schedule says
"articles for the conveyance or packing of goods". Either alone loses
half the candidates.

Why RRF rather than averaging the scores? BM25 scores and cosine
similarities are on different scales and are not comparable. RRF uses
only each engine's *rank*, so no normalisation is needed. That is exactly
what EnsembleRetriever implements, which is why it is used here instead
of a custom fusion function.

This stage optimises recall, not precision: the paper reports 22% of its
errors were candidates never retrieved at all. The two LLM ranking stages
that follow do the narrowing.

Everything runs through LangChain Runnables, so each retrieval shows up
in LangSmith as its own step with its inputs and outputs.
"""

from __future__ import annotations

import csv
import functools
import re
from pathlib import Path

from langchain_core.documents import Document

TARIFF_CSV = Path("data/processed/tariff.csv")
DEFAULT_K = 30
# Weights favour the lexical side slightly: HS descriptions are precise
# technical text, so an exact term match is usually the stronger signal.
ENSEMBLE_WEIGHTS = [0.6, 0.4]


def _import_ensemble_retriever():
    """Locate EnsembleRetriever across LangChain versions.

    It lives in `langchain.retrievers` up to 0.3.x and moved to
    `langchain_classic.retrievers` in 1.x. Trying both keeps a fresh
    `pip install` working whichever the resolver picks, instead of
    failing on a machine with a different Python version.
    """
    for module in ("langchain_classic.retrievers", "langchain.retrievers"):
        try:
            return __import__(module, fromlist=["EnsembleRetriever"]).EnsembleRetriever
        except (ImportError, AttributeError):
            continue
    return None


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, digits kept.

    Digits matter: an importer often types part of a code ("3901 black"),
    and a tokenizer that strips numbers makes that query unanswerable.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def load_rows(path: Path = TARIFF_CSV) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_to_documents(rows: list[dict]) -> list[Document]:
    return [
        Document(
            page_content=f"{r['hs_code']} {r['description']}",
            metadata={k: r[k] for k in r},
        )
        for r in rows
    ]


def doc_to_dict(doc: Document) -> dict:
    """Flatten a Document back into the plain dict the agents expect."""
    return {"text": doc.page_content, **doc.metadata}


class HybridRetriever:
    """Wraps LangChain retrievers behind one `search` call.

    Degrades on purpose: if embeddings or Chroma are unavailable it runs
    BM25 alone rather than failing, so the pipeline still works before
    the vector store has been indexed.
    """

    def __init__(self, csv_path: Path = TARIFF_CSV, k: int = DEFAULT_K):
        self.rows = load_rows(csv_path)
        self.documents = rows_to_documents(self.rows)
        self.k = k
        # Constructing a dense retriever always succeeds; it only reaches
        # the network on the first query. So availability has to be
        # decided at query time, and remembered once it fails.
        self._dense_broken = False

    # ------------------------------------------------------------ lexical
    @functools.cached_property
    def bm25(self):
        if not self.documents:
            return None
        from langchain_community.retrievers import BM25Retriever

        # The default preprocessor is str.split, which leaves case and
        # punctuation attached: "Polypropylene," never matches the query
        # "polypropylene". Our tokenizer lowercases and keeps digits, so
        # HS code fragments like "3903" stay searchable too.
        retriever = BM25Retriever.from_documents(
            self.documents, preprocess_func=tokenize
        )
        retriever.k = self.k
        return retriever

    # -------------------------------------------------------------- dense
    @functools.cached_property
    def dense(self):
        try:
            from backend.rag.store import TARIFF_COLLECTION, get_vectorstore

            return get_vectorstore(TARIFF_COLLECTION).as_retriever(
                search_kwargs={"k": self.k}
            )
        except Exception as exc:
            print(f"[retriever] dense retrieval unavailable, BM25 only: {exc}")
            return None

    # ----------------------------------------------------------- ensemble
    @functools.cached_property
    def ensemble(self):
        if self.bm25 is None:
            return None
        if self.dense is None:
            return self.bm25

        EnsembleRetriever = _import_ensemble_retriever()
        if EnsembleRetriever is None:
            print("[retriever] EnsembleRetriever unavailable, BM25 only")
            return self.bm25

        return EnsembleRetriever(
            retrievers=[self.bm25, self.dense], weights=ENSEMBLE_WEIGHTS
        )

    # ------------------------------------------------------------- public
    def search(self, query: str, k: int | None = None,
               chapter: str | None = None) -> list[dict]:
        """Retrieve candidate tariff lines for a product description.

        Falls back to BM25 if the dense side fails — an expired embedding
        model, a revoked key, an offline machine. Degraded retrieval is a
        worse answer; a raised exception is no answer at all, and this
        runs mid-pipeline with a user waiting.
        """
        limit = k or self.k

        if not self._dense_broken and self.ensemble is not None:
            try:
                docs = self.ensemble.invoke(query)
                return self._finish(docs, limit, chapter)
            except Exception as exc:
                # Remember it: retrying on every query would add a network
                # timeout to every line item on the invoice.
                self._dense_broken = True
                print(f"[retriever] dense retrieval failed, BM25 only from here: {exc}")

        return self.bm25_search(query, k=limit, chapter=chapter)

    def _finish(self, docs, limit: int, chapter: str | None) -> list[dict]:
        if chapter:
            docs = [d for d in docs if d.metadata.get("chapter") == chapter]
        return [doc_to_dict(d) for d in docs[:limit]]

    def bm25_search(self, query: str, k: int | None = None,
                    chapter: str | None = None) -> list[dict]:
        """Lexical-only search. Used by tests and as the offline path."""
        if self.bm25 is None:
            return []
        return self._finish(self.bm25.invoke(query), k or self.k, chapter)

    @property
    def mode(self) -> str:
        """Which retrieval path is live. Surfaced in /health and the UI so
        a silent downgrade to BM25 is visible rather than mysterious."""
        if self.bm25 is None:
            return "unavailable"
        if self._dense_broken or self.dense is None:
            return "bm25_only"
        return "hybrid"


_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
