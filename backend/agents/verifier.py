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
def verify(description: str, candidates: list[HSCandidate]) -> Verdict:
    """Adversarially review the chosen classification."""
    if not candidates:
        return Verdict(
            verdict="fail", code_matches=False, needs_review=True,
            warnings=["No tariff heading could be matched to this item. "
                      "It may fall outside the loaded schedule."],
        )

    chosen = candidates[0]
    threshold = get_settings().confidence_threshold

    # A cheap deterministic gate before spending an LLM call.
    if chosen.confidence < threshold:
        return Verdict(
            verdict="fail", confidence_justified=False, needs_review=True,
            warnings=[
                f"Confidence {chosen.confidence:.0%} is below the "
                f"{threshold:.0%} threshold. A broker should confirm "
                f"{chosen.hs_code} before this is filed."
            ],
        )

    evidence_text = "\n".join(
        f"  [{e.source}] \"{e.quote}\"" for e in chosen.evidence
    ) or "  (no notes were cited)"
    alternatives = "\n".join(
        f"  {c.hs_code} ({c.confidence:.0%}) {c.heading_text}"
        for c in candidates[1:]
    ) or "  (none)"

    try:
        result = structured("judge", Verdict).invoke([
            ("system", VERIFY_SYSTEM),
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
            warnings=[f"Verification could not be completed ({exc}). "
                      "Treat this classification as unconfirmed."],
        )

    # A close runner-up is a genuine ambiguity even when the judge is happy.
    if len(candidates) > 1 and (chosen.confidence - candidates[1].confidence) < 0.10:
        result.needs_review = True
        result.warnings.append(
            f"{candidates[1].hs_code} scores almost as well as "
            f"{chosen.hs_code}. Both are defensible; the choice changes the duty."
        )

    if not chosen.evidence:
        result.warnings.append(
            "No chapter note was cited for this heading, so the "
            "classification rests on description matching alone."
        )

    return result
