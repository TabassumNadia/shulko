"""
Classifier agent — three-stage HS classification.

Implements the deterministic workflow of arXiv 2605.14857 in reduced form:

    Stage 1  hybrid retrieval        ~30 candidates, recall first
    Stage 2  LLM shortlist            ~8, on heading text alone
    Stage 3  LLM rank WITH the notes  top 3, with verbatim citations

Two decisions worth defending on camera:

Why not one LLM call? Because a single call that receives 30 candidates
and every chapter note ignores the notes. Forcing a commitment to a
shortlist first, then re-reading it against the law, is what makes the
exclusion clauses bite.

Why return three codes and not one? Because that paper's error analysis
found 65% of its mistakes were ranking errors on candidates that HAD been
retrieved — the right answer was in the list, just not first. Showing the
top three surfaces the real answer far more often than showing one, and
it tells the user where the system is unsure.
"""

from __future__ import annotations

from langsmith import traceable
from pydantic import BaseModel, Field

from backend.core.llm import structured
from backend.core.schemas import Evidence, HSCandidate, LineItem
from backend.prompts.classify import (
    RANK_SYSTEM, RANK_USER, SHORTLIST_SYSTEM, SHORTLIST_USER,
)
from backend.rag.retriever import get_retriever

RETRIEVE_K = 30
SHORTLIST_K = 8


# --- structured output shapes -------------------------------------------
class Shortlist(BaseModel):
    hs_codes: list[str] = Field(description="Codes from the candidate list only")


class RankedCandidate(BaseModel):
    hs_code: str
    heading_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    gir_rule: str = Field(default="", description="GIR rule applied, e.g. '3(a)'")
    evidence: list[Evidence] = Field(default_factory=list)


class Ranking(BaseModel):
    candidates: list[RankedCandidate]


# --- stages --------------------------------------------------------------
@traceable(name="02a_retrieve_candidates", run_type="retriever")
def retrieve(item: LineItem) -> list[dict]:
    """Stage 1 — hybrid BM25 + dense retrieval, tuned for recall.

    The query blends the raw description with the extracted attributes,
    because the schedule's wording rarely matches an invoice's wording.
    """
    a = item.attributes
    query = " ".join(filter(None, [
        item.description, a.material, a.form, a.function,
        *a.distinguishing_features,
    ]))
    return get_retriever().search(query, k=RETRIEVE_K)


@traceable(name="02b_shortlist_headings", run_type="chain")
def shortlist(item: LineItem, candidates: list[dict]) -> list[dict]:
    """Stage 2 — narrow to ~8 on heading text alone, no notes yet."""
    if len(candidates) <= SHORTLIST_K:
        return candidates

    listing = "\n".join(
        f"  {c['hs_code']}  {c['description']}" for c in candidates
    )
    a = item.attributes
    model = structured("reasoning", Shortlist)
    picked = model.invoke([
        ("system", SHORTLIST_SYSTEM),
        ("user", SHORTLIST_USER.format(
            description=item.description,
            material=a.material or "unknown",
            form=a.form or "unknown",
            function=a.function or "unknown",
            features=", ".join(a.distinguishing_features) or "none",
            candidates=listing,
        )),
    ])

    # Never trust the model to stay inside the candidate set.
    by_code = {c["hs_code"]: c for c in candidates}
    kept = [by_code[c] for c in picked.hs_codes if c in by_code]
    return kept[:SHORTLIST_K] or candidates[:SHORTLIST_K]


@traceable(name="02c_rank_with_chapter_notes", run_type="chain")
def rank_with_notes(item: LineItem, candidates: list[dict]) -> list[HSCandidate]:
    """Stage 3 — re-rank against the Section and Chapter Notes.

    This is the stage that can demote a heading. Everything before it is
    similarity; this is law.
    """
    chapters = sorted({c["chapter"] for c in candidates})
    notes = _load_notes(chapters)

    # Count the notes per chapter, so each returned candidate can record
    # whether the law governing ITS OWN chapter was in front of the model.
    # A single total across all candidate chapters would credit a
    # chapter-52 candidate with chapter-61's notes and hide exactly the
    # distinction the Verifier needs.
    notes_per_chapter: dict[str, int] = {}
    for n in notes:
        chapter = n.get("chapter", "")
        notes_per_chapter[chapter] = notes_per_chapter.get(chapter, 0) + 1

    listing = "\n".join(
        f"  {c['hs_code']}  {c['description']}" for c in candidates
    )
    notes_text = "\n\n".join(
        f"[{n['source']}] {n['text']}" for n in notes
    ) or "No notes available for these chapters."

    a = item.attributes
    model = structured("reasoning", Ranking)
    ranked = model.invoke([
        ("system", RANK_SYSTEM),
        ("user", RANK_USER.format(
            description=item.description,
            material=a.material or "unknown",
            form=a.form or "unknown",
            function=a.function or "unknown",
            candidates=listing,
            notes=notes_text,
        )),
    ])

    by_code = {c["hs_code"]: c for c in candidates}
    out: list[HSCandidate] = []
    for r in ranked.candidates[:3]:
        if r.hs_code not in by_code:
            continue        # hallucinated code, drop it silently
        out.append(HSCandidate(
            hs_code=r.hs_code,
            heading_text=by_code[r.hs_code]["description"],
            confidence=r.confidence,
            reasoning=(f"[GIR {r.gir_rule}] " if r.gir_rule else "") + r.reasoning,
            evidence=r.evidence,
            notes_available=notes_per_chapter.get(
                by_code[r.hs_code]["chapter"], 0),
        ))
    return out


@traceable(name="02_classify_hs_code", run_type="chain")
def classify(item: LineItem) -> list[HSCandidate]:
    """Run all three stages. Returns up to 3 candidates, best first."""
    candidates = retrieve(item)
    if not candidates:
        return []

    short = shortlist(item, candidates)
    ranked = rank_with_notes(item, short)

    if ranked:
        return ranked

    # Retrieval found something but ranking produced nothing usable.
    # Fall back to the top retrieved line at low confidence rather than
    # returning nothing — and let the Verifier flag it.
    top = short[0]
    return [HSCandidate(
        hs_code=top["hs_code"],
        heading_text=top["description"],
        confidence=0.25,
        reasoning="Retrieval only; the ranking stage returned no usable candidate.",
        notes_available=len(_load_notes([top["chapter"]])),
    )]


def _load_notes(chapters: list[str]) -> list[dict]:
    """All notes for the candidate chapters — not a top-k search.

    The classifier needs every note that could exclude a candidate, so
    this reads them whole rather than retrieving the most similar few.
    """
    return _notes_by_chapter().get_many(chapters)


class _NoteCache:
    """Chapter notes, loaded once per process.

    There are 77 of them and they never change at runtime. Fetching them
    from Chroma on every line item, for every candidate chapter, was
    costing more time than the LLM calls they feed.
    """

    def __init__(self) -> None:
        self._by_chapter: dict[str, list[dict]] | None = None

    def _load(self) -> dict[str, list[dict]]:
        notes: list[dict] = []
        try:
            from backend.rag.store import NOTES_COLLECTION, get_vectorstore

            got = get_vectorstore(NOTES_COLLECTION).get()
            notes = [
                {"text": text, **meta}
                for text, meta in zip(got.get("documents", []),
                                      got.get("metadatas", []))
            ]
        except Exception:
            notes = []

        if not notes:
            # Chroma not indexed yet — read the JSONL so the classifier
            # still applies the law rather than silently skipping it.
            import json
            from pathlib import Path

            path = Path("data/processed/notes.jsonl")
            if path.exists():
                notes = [json.loads(l) for l in
                         path.open(encoding="utf-8") if l.strip()]

        grouped: dict[str, list[dict]] = {}
        for n in notes:
            grouped.setdefault(n.get("chapter", "??"), []).append(n)
        return grouped

    def _ensure(self) -> dict[str, list[dict]]:
        if self._by_chapter is None:
            self._by_chapter = self._load()
        return self._by_chapter

    def get_many(self, chapters: list[str]) -> list[dict]:
        loaded = self._ensure()
        out: list[dict] = []
        for ch in chapters:
            out.extend(loaded.get(ch, []))
        return out

    def chapters(self) -> list[str]:
        return sorted(self._ensure())


_note_cache = _NoteCache()


def _notes_by_chapter() -> _NoteCache:
    return _note_cache


def chapters_with_notes() -> list[str]:
    """Chapters the knowledge base actually holds Section/Chapter Notes for.

    The Verifier reports this when nothing could be cited, so the message
    names a real coverage limit ("no note is loaded for Chapter 54")
    instead of sounding like a retrieval failure. Read from the same cache
    the classifier uses, so the two can never disagree.
    """
    return _note_cache.chapters()
