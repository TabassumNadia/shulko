"""
Verifier agent -- the component that makes this system auditable.

arXiv 2605.06635 measured deep research agents and found citations that
resolve (94-100%) and look relevant (80%+) while the underlying facts are
right only 39-77% of the time. Surface citation quality hides factual
failure. A pipeline that stops at "it cited something" inherits that gap.

So the last node's only job is to disagree: check the code against the
goods, check each quote against the claim built on it, and check whether
the stated confidence is earned. Any failure sets needs_review and the
number goes to a human before anyone pays it.
"""

from __future__ import annotations

from langsmith import traceable
from pydantic import BaseModel, Field

from backend.core.config import get_settings
from backend.core.i18n import language_directive, tr
from backend.core.llm import structured
from backend.core.schemas import HSCandidate
from backend.prompts.verify import VERIFY_SYSTEM, VERIFY_USER


class Verdict(BaseModel):
    verdict: str = Field(description="'pass' or 'fail'")
    code_matches: bool = True
    evidence_supported: bool = True
    confidence_justified: bool = True
    needs_review: bool = False
    warnings: list[str] = Field(default_factory=list)


@traceable(name="05_verify_classification", run_type="chain")
def verify(
    description: str, candidates: list[HSCandidate], language: str = "en"
) -> Verdict:
    """Adversarially review the chosen classification."""
    if not candidates:
        return Verdict(
            verdict="fail", code_matches=False, needs_review=True,
            warnings=[tr("no_heading_could_be_matched", language)],
        )

    chosen = candidates[0]
    threshold = get_settings().confidence_threshold

    # A cheap deterministic gate before spending an LLM call.
    if chosen.confidence < threshold:
        return Verdict(
            verdict="fail", confidence_justified=False, needs_review=True,
            warnings=[tr(
                "confidence_below_threshold", language,
                confidence=f"{chosen.confidence:.0%}",
                threshold=f"{threshold:.0%}", hs_code=chosen.hs_code,
            )],
        )

    evidence_text = "\n".join(
        f"  [{e.source}] \"{e.quote}\"" for e in chosen.evidence
    ) or f"  ({_absence_reason(chosen)})"
    alternatives = "\n".join(
        f"  {c.hs_code} ({c.confidence:.0%}) {c.heading_text}"
        for c in candidates[1:]
    ) or "  (none)"

    try:
        result = structured("judge", Verdict).invoke([
            ("system", VERIFY_SYSTEM + language_directive(language)),
            ("user", VERIFY_USER.format(
                description=description,
                hs_code=chosen.hs_code,
                heading_text=chosen.heading_text,
                confidence=f"{chosen.confidence:.0%}",
                reasoning=chosen.reasoning,
                evidence=evidence_text,
                alternatives=alternatives,
            )),
        ])
    except Exception as exc:
        # If the judge itself fails, flag rather than pass silently.
        return Verdict(
            verdict="fail", needs_review=True,
            warnings=[tr("verification_failed", language, error=exc)],
        )

    # A close runner-up is a genuine ambiguity even when the judge is happy.
    if len(candidates) > 1 and (chosen.confidence - candidates[1].confidence) < 0.10:
        result.needs_review = True
        result.warnings.append(tr(
            "close_runner_up", language,
            other_code=candidates[1].hs_code, hs_code=chosen.hs_code,
        ))

    if not chosen.evidence:
        result.warnings.append(_absence_warning(chosen, language))
        if chosen.notes_available:
            # The governing notes WERE in front of the model and it
            # answered without quoting any of them. That is a gap in the
            # reasoning, not in the knowledge base, so it goes to a human.
            # The other branch is a known coverage limit and must not set
            # this, or every item outside the indexed chapters would be
            # flagged for something no reviewer can fix.
            result.needs_review = True

    return result


def _chapter_of(candidate: HSCandidate) -> str:
    """The two-digit chapter an HS code sits in."""
    return candidate.hs_code[:2]


def _absence_reason(candidate: HSCandidate) -> str:
    """Why no note is quoted — phrased for the judge, not the importer.

    The judge scores evidence support. Told only "no notes were cited" it
    cannot tell a classifier that ignored the law from one that had no law
    to read, and it marks both down the same. Naming the reason is what
    keeps check 2 meaningful.
    """
    chapter = _chapter_of(candidate)
    if candidate.notes_available:
        return (
            f"none quoted, although {candidate.notes_available} Section/Chapter "
            f"Note(s) for Chapter {chapter} WERE supplied to the classifier"
        )
    return (
        "none quoted, and none exists to quote: the knowledge base holds no "
        f"Section or Chapter Note for Chapter {chapter}"
    )


def _absence_warning(candidate: HSCandidate, language: str = "en") -> str:
    """The same distinction, phrased for the importer.

    Two different facts used to share one sentence. "No chapter note was
    cited" reads as a failure, which is right when a note existed and was
    skipped and misleading when the schedule chapter has no note loaded at
    all — the importer cannot act on the second and should not be alarmed
    by it.
    """
    chapter = _chapter_of(candidate)
    if candidate.notes_available:
        return tr(
            "notes_available_not_cited", language,
            notes_available=candidate.notes_available, chapter=chapter,
            hs_code=candidate.hs_code,
        )

    try:
        from backend.agents.classifier import chapters_with_notes
        covered = ", ".join(chapters_with_notes())
    except Exception:
        covered = ""

    return tr(
        "no_note_loaded_for_chapter", language,
        chapter=chapter, hs_code=candidate.hs_code,
        coverage=tr("notes_coverage_suffix", language, covered=covered) if covered else "",
    )
