"""Pydantic models shared across agents, tools and the API."""

from typing import Literal
from pydantic import BaseModel, Field


class ItemAttributes(BaseModel):
    """Structured attributes extracted before any tariff lookup.

    Mirrors Stage 1 of the deterministic workflow in arXiv 2605.14857.
    """
    material: str | None = None
    form: str | None = None
    function: str | None = None
    distinguishing_features: list[str] = Field(default_factory=list)


class LineItem(BaseModel):
    description: str
    quantity: float = 1.0
    unit_price: float = 0.0
    currency: str = "USD"
    origin_country: str | None = None
    attributes: ItemAttributes = Field(default_factory=ItemAttributes)


class Evidence(BaseModel):
    """A verbatim quote that justifies a classification."""
    source: str                 # e.g. "Chapter 39, Note 2(b)"
    quote: str
    supports: bool = True


class HSCandidate(BaseModel):
    hs_code: str
    heading_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    # How many Section/Chapter Notes for THIS candidate's own chapter were
    # placed in front of the ranking model.
    #
    # Without this the Verifier cannot tell two very different situations
    # apart, because both arrive as an empty `evidence` list:
    #   0   the knowledge base holds no note for this chapter, so there
    #       was nothing to quote and silence is honest;
    #   > 0 the model was shown the governing law and did not quote it,
    #       which is a real evidence gap and belongs in front of a human.
    notes_available: int = 0


class Advisory(BaseModel):
    """A grounded regulatory note — never stated without its source."""
    kind: Literal["sro", "restriction", "certificate", "info"] = "info"
    message: str
    source_url: str


class LineItemResult(BaseModel):
    item: LineItem
    candidates: list[HSCandidate] = Field(default_factory=list)
    chosen_hs_code: str | None = None
    duty: dict | None = None
    # Statutory rates as percentages. A question with no invoice attached
    # has no value to compute on, so the honest answer is the rate card,
    # not a taka figure derived from a price of zero.
    rates_pct: dict | None = None
    advisories: list[Advisory] = Field(default_factory=list)
    needs_review: bool = False
    warnings: list[str] = Field(default_factory=list)
