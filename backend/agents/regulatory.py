"""
Regulatory agent -- live Google Search grounding.

Rates come from the tariff schedule, which is a snapshot. Reality moves:
NBR issues SROs, goods get restricted, certificates become mandatory. That
gap is what this agent closes, and it is why the system needs the internet
rather than a bigger PDF.

The discipline here is refusal. A claim without a retrieved source is
dropped, not softened -- because an importer may act on it.
"""

from __future__ import annotations

import re

from langsmith import traceable

from backend.core.schemas import Advisory
from backend.prompts.regulatory import REGULATORY_SYSTEM, REGULATORY_USER
from backend.tools.grounded_search import grounded_answer

MAX_ADVISORIES = 4

KINDS = [
    ("sro", re.compile(r"\bSRO\b|statutory regulatory order|প্রজ্ঞাপন", re.I)),
    ("restriction", re.compile(r"\b(ban|banned|prohibit|restrict)", re.I)),
    ("certificate", re.compile(r"certificat|BSTI|radiation|phytosanitary|BTRC|permit|licen[cs]e", re.I)),
]


def _classify(text: str) -> str:
    for kind, pattern in KINDS:
        if pattern.search(text):
            return kind
    return "info"


def _split_claims(text: str) -> list[str]:
    """One advisory per bullet or sentence, so each can carry its source."""
    lines = [l.strip(" -*•\t") for l in text.splitlines() if l.strip()]
    bullets = [l for l in lines if len(l) > 25]
    if len(bullets) > 1:
        return bullets
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 25]


@traceable(name="04_regulatory_check", run_type="chain")
def check_regulations(hs_code: str, description: str) -> list[Advisory]:
    """Search for live requirements affecting this code.

    Returns an empty list when nothing relevant was retrieved. Empty is a
    valid, useful answer -- it means no live obligation was found, not
    that the agent failed.
    """
    answer = grounded_answer(
        REGULATORY_SYSTEM,
        REGULATORY_USER.format(hs_code=hs_code, description=description),
        # A literal search engine needs a search string, not a prompt.
        query=f"Bangladesh import {description} HS {hs_code} "
              f"NBR SRO restriction certificate requirement",
    )

    # No sources means nothing was actually retrieved. Whatever the model
    # wrote came from memory, so it does not ship.
    if not answer.is_grounded or not answer.text:
        return []

    primary = answer.sources[0]["url"]
    advisories: list[Advisory] = []
    for claim in _split_claims(answer.text)[:MAX_ADVISORIES]:
        # Attach a source named in the claim if one matches; else the
        # top retrieved source.
        url = next(
            (s["url"] for s in answer.sources
             if s["url"] in claim or s.get("title", "")[:20] in claim),
            primary,
        )
        advisories.append(Advisory(
            kind=_classify(claim), message=claim, source_url=url
        ))
    return advisories
