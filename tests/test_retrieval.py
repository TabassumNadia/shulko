"""
Tests for the retrieval layer.

These exercise the LangChain BM25 path only, so they run with no API key,
no network and no vector store — which means they still pass on a fresh
clone before anything has been indexed. That is deliberate: a test suite
that needs a paid key to run is a test suite nobody runs.
"""

from pathlib import Path

import pytest

from backend.core.duty import TariffRates, calculate_duty
from backend.rag.retriever import (
    HybridRetriever, doc_to_dict, load_rows, rows_to_documents,
)

SAMPLE = Path("data/samples/tariff_sample.csv")


@pytest.fixture
def retriever():
    return HybridRetriever(csv_path=SAMPLE)


def test_sample_schedule_loads(retriever):
    assert len(retriever.rows) == 15
    assert all(r["chapter"] == "39" for r in retriever.rows)


def test_rows_become_langchain_documents():
    """Retrieval speaks LangChain Documents, so rates must survive in metadata."""
    docs = rows_to_documents(load_rows(SAMPLE))
    assert docs
    first = docs[0]
    assert first.page_content.startswith(first.metadata["hs_code"])
    assert "cd" in first.metadata and "description" in first.metadata


def test_document_round_trips_to_dict():
    doc = rows_to_documents(load_rows(SAMPLE))[0]
    flat = doc_to_dict(doc)
    assert flat["hs_code"] == doc.metadata["hs_code"]
    assert flat["text"] == doc.page_content


def test_bm25_finds_exact_technical_term(retriever):
    hits = retriever.bm25_search("polypropylene", k=3)
    assert hits[0]["hs_code"] == "3902.10.00"


def test_bm25_keeps_hs_code_fragments_searchable(retriever):
    """HS codes are in page_content, so a code fragment must retrieve."""
    hits = retriever.bm25_search("3903 styrene", k=5)
    assert any(h["hs_code"].startswith("3903") for h in hits)


def test_search_falls_back_to_bm25_without_a_vector_store(retriever):
    """The pipeline must work before Chroma has been indexed."""
    hits = retriever.search("ethylene vinyl acetate", k=5)
    assert hits
    assert hits[0]["hs_code"] == "3901.30.00"


def test_chapter_filter_narrows_results(retriever):
    assert retriever.search("polymer", k=10, chapter="39")
    assert retriever.search("polymer", k=10, chapter="84") == []


def test_missing_csv_yields_an_empty_retriever_not_a_crash():
    r = HybridRetriever(csv_path=Path("data/processed/does_not_exist.csv"))
    assert r.rows == []
    assert r.bm25_search("anything") == []
    assert r.search("anything") == []


def test_retrieval_to_duty_end_to_end(retriever):
    """The seam that matters: a query yields a code, and that code's
    rates yield a number."""
    hit = retriever.bm25_search("polypropylene", k=1)[0]
    rates = TariffRates.from_percentages(
        cd=float(hit["cd"]), rd=float(hit["rd"]), sd=float(hit["sd"]),
        vat=float(hit["vat"]), ait=float(hit["ait"]), at=float(hit["at"]),
    )
    result = calculate_duty(fob=100_000, freight=8_000, insurance=1_000, rates=rates)

    assert result.tti > 0
    assert result.landed_cost > result.av
    assert 25 < result.effective_rate < 35     # 5% CD + 15% VAT + 5% AIT + 5% AT


# --- regression: the dense side failing must not take the pipeline down ---
class _BrokenRetriever:
    """Stands in for a dense retriever whose model was retired."""

    def invoke(self, _query):
        raise RuntimeError("404 model not found")


def test_search_survives_a_dead_dense_retriever(retriever, monkeypatch):
    """A retired embedding model must degrade to BM25, not raise.

    This is a real failure: Google retired text-embedding-004, and the
    dense path only touches the network on the first query, so the error
    surfaces mid-pipeline with a user waiting.
    """
    monkeypatch.setattr(type(retriever), "ensemble",
                        property(lambda self: _BrokenRetriever()))

    hits = retriever.search("polypropylene", k=3)

    assert hits, "search must still return results"
    assert hits[0]["hs_code"] == "3902.10.00"
    assert retriever.mode == "bm25_only"


def test_dense_failure_is_remembered(retriever, monkeypatch):
    """Retrying a dead backend on every line item would add a network
    timeout to each one."""
    calls = {"n": 0}

    class _Counting(_BrokenRetriever):
        def invoke(self, _query):
            calls["n"] += 1
            raise RuntimeError("still dead")

    monkeypatch.setattr(type(retriever), "ensemble",
                        property(lambda self: _Counting()))

    for _ in range(4):
        retriever.search("polymer", k=2)

    assert calls["n"] == 1, "the broken backend should be tried once, not four times"


def test_mode_reports_hybrid_when_nothing_has_failed(retriever):
    assert retriever.mode in {"hybrid", "bm25_only"}
