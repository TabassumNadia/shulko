"""
Tests for the two defects fixed in this change.

1. Chapter Notes DID reach the classifier, but the Verifier could not tell
   an uncited note from a note that does not exist. Both arrived as an
   empty evidence list, so every item outside the indexed chapters drew a
   warning that read like a retrieval failure.

2. The grounding step reported an empty advisory list both when the web
   was searched and held nothing, and when no search could run at all.

None of these tests need a network or an API key.
"""

from __future__ import annotations

import pytest

from backend.agents.verifier import (
    Verdict, _absence_reason, _absence_warning, _chapter_of, verify,
)
from backend.core.schemas import Evidence, HSCandidate


class _PassingJudge:
    """A judge that waves everything through.

    The point of these tests is what the Verifier adds on top of the
    judge's opinion, so the judge must not be the thing under test.
    """

    def invoke(self, _messages):
        return Verdict(verdict="pass", needs_review=False)


@pytest.fixture
def passing_judge(monkeypatch):
    import backend.agents.verifier as verifier

    monkeypatch.setattr(verifier, "structured", lambda *a, **k: _PassingJudge())


# ----------------------------------------------------- note provenance
def test_chapter_is_read_from_the_code_not_guessed():
    assert _chapter_of(HSCandidate(
        hs_code="6109.10.00", heading_text="t-shirts", confidence=0.9)) == "61"


def test_an_uncited_available_note_is_reported_as_an_evidence_gap():
    """The notes were in front of the model and it quoted none of them."""
    warning = _absence_warning(HSCandidate(
        hs_code="6109.10.00", heading_text="T-shirts, of cotton",
        confidence=0.95, notes_available=9,
    ))
    assert "9" in warning
    assert "Chapter 61" in warning
    assert "available" in warning.lower()
    # Must not claim the knowledge base is empty, because it is not.
    assert "no section or chapter note for chapter" not in warning.lower()


def test_a_chapter_with_no_notes_is_reported_as_a_coverage_limit():
    """Nothing existed to quote, so blaming the classifier would mislead."""
    warning = _absence_warning(HSCandidate(
        hs_code="5404.12.00", heading_text="Monofilament",
        confidence=0.95, notes_available=0,
    ))
    assert "Chapter 54" in warning
    assert "no section or chapter note" in warning.lower()
    assert "but none was cited" not in warning


def test_the_two_absences_never_produce_the_same_sentence():
    available = HSCandidate(hs_code="6109.10.00", heading_text="x",
                            confidence=0.9, notes_available=9)
    absent = HSCandidate(hs_code="5404.12.00", heading_text="x",
                         confidence=0.9, notes_available=0)
    assert _absence_warning(available) != _absence_warning(absent)
    assert _absence_reason(available) != _absence_reason(absent)


def test_the_judge_is_told_why_the_evidence_is_missing():
    """The judge scores evidence support, so it must not mark down an
    absence the knowledge base made unavoidable."""
    assert "none exists to quote" in _absence_reason(HSCandidate(
        hs_code="5404.12.00", heading_text="x", confidence=0.9,
        notes_available=0))
    assert "WERE supplied" in _absence_reason(HSCandidate(
        hs_code="6109.10.00", heading_text="x", confidence=0.9,
        notes_available=9))


def test_an_uncited_available_note_forces_human_review(passing_judge):
    """The strengthening half: evidence that was there and went unused is
    a real gap, so it must not pass silently."""
    verdict = verify("cotton t-shirts", [HSCandidate(
        hs_code="6109.10.00", heading_text="T-shirts, of cotton",
        confidence=0.95, notes_available=9,
    )])
    assert verdict.needs_review is True
    assert any("none was cited" in w for w in verdict.warnings)


def test_a_chapter_without_notes_does_not_force_review(passing_judge):
    """The other half: a known coverage limit must not flag every item for
    something no reviewer can fix."""
    verdict = verify("plastic monofilament", [HSCandidate(
        hs_code="5404.12.00", heading_text="Monofilament",
        confidence=0.95, notes_available=0,
    )])
    assert verdict.needs_review is False
    assert any("Chapter 54" in w for w in verdict.warnings)


def test_a_cited_note_produces_no_absence_warning(passing_judge):
    verdict = verify("cotton t-shirts", [HSCandidate(
        hs_code="6109.10.00", heading_text="T-shirts, of cotton",
        confidence=0.95, notes_available=9,
        evidence=[Evidence(source="Chapter 61, Note 1",
                           quote="This Chapter applies only to made up "
                                 "knitted or crocheted articles.")],
    )])
    assert verdict.needs_review is False
    assert not any("cited" in w for w in verdict.warnings)


# ------------------------------------------- notes really reach ranking
def test_the_note_cache_reports_the_chapters_it_holds():
    from backend.agents.classifier import chapters_with_notes

    chapters = chapters_with_notes()
    assert chapters == sorted(chapters)
    # A regression in ingest or in the cache would empty this and silently
    # disable every citation in the system.
    assert {"39", "61"} <= set(chapters)


def test_notes_are_loaded_for_a_chapter_that_has_them():
    from backend.agents.classifier import _load_notes

    notes = _load_notes(["61"])
    assert notes, "Chapter 61 notes must be retrievable at runtime"
    assert any("knitted" in n["text"].lower() for n in notes)


def test_notes_are_empty_for_a_chapter_the_corpus_does_not_cover():
    from backend.agents.classifier import _load_notes

    assert _load_notes(["54"]) == []


def test_a_candidate_is_credited_only_with_its_own_chapters_notes():
    """A chapter-52 candidate must not inherit chapter-61's notes."""
    from backend.agents.classifier import _load_notes

    per_chapter: dict[str, int] = {}
    for note in _load_notes(["52", "61"]):
        per_chapter[note["chapter"]] = per_chapter.get(note["chapter"], 0) + 1

    assert per_chapter.get("61", 0) > 0
    assert per_chapter.get("52", 0) == 0


# ------------------------------------------------------ grounding honesty
def test_not_searched_is_distinguishable_from_nothing_found():
    from backend.agents.regulatory import RegulatoryResult

    could_not_search = RegulatoryResult(advisories=[], mechanism="none")
    searched_found_nothing = RegulatoryResult(advisories=[],
                                              mechanism="google_search")

    assert could_not_search.searched is False
    assert searched_found_nothing.searched is True
    # Both carry no advisories. The mechanism is the only thing telling
    # them apart, which is exactly why it is carried.
    assert could_not_search.advisories == searched_found_nothing.advisories


@pytest.mark.parametrize("mechanism", ["google_search", "tavily"])
def test_a_search_that_ran_is_reported_as_having_run(mechanism):
    from backend.agents.regulatory import RegulatoryResult

    assert RegulatoryResult(mechanism=mechanism).searched is True
