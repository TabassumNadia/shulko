"""
Tests for partial failure.

A long invoice makes many model calls, and on a free-tier key some of
them will fail. The design decision under test is that a failure is
*contained*: the item that failed says so, and every other item still
carries its number.

The alternative — one exception ending the run — is what the API used to
do, and it turned a recoverable hiccup on line 2 into a 500 that threw
away lines 1 and 3 as well.
"""

from __future__ import annotations

import backend.graph.builder as builder


def _items(*descriptions):
    return [
        {"description": d, "quantity": 1.0, "unit_price": 100.0,
         "currency": "USD", "origin_country": None,
         "attributes": {"material": None, "form": None, "function": None,
                        "distinguishing_features": []}}
        for d in descriptions
    ]


def test_one_failing_item_does_not_lose_the_others(monkeypatch):
    """The classifier blows up on the second item only."""

    def flaky_classify(item, language="en"):
        if "explodes" in item.description:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return []

    monkeypatch.setattr("backend.agents.classifier.classify", flaky_classify)

    state = {"line_items": _items("good one", "explodes", "good two"),
             "importer_type": "commercial"}
    results = builder.classify_and_cost_node(state)["results"]

    assert len(results) == 3, "every line item must come back"

    failed = results[1]
    assert failed["needs_review"] is True
    assert any("Classification failed" in w for w in failed["warnings"])

    # The neighbours are untouched by their neighbour's failure.
    for ok in (results[0], results[2]):
        assert not any("Classification failed" in w for w in ok["warnings"])


def test_a_failed_item_is_flagged_rather_than_silently_dropped(monkeypatch):
    """Silence would be the dangerous outcome: a missing line reads as a
    line with no duty, and the importer under-declares."""

    monkeypatch.setattr(
        "backend.agents.classifier.classify",
        lambda item, language="en": (_ for _ in ()).throw(RuntimeError("boom")),
    )

    state = {"line_items": _items("anything"), "importer_type": "commercial"}
    result = builder.classify_and_cost_node(state)["results"][0]

    assert result["needs_review"] is True
    assert result["duty"] is None
    assert result["warnings"], "a failure must be stated, not implied"


def test_regulatory_failure_leaves_the_duty_figures_intact(monkeypatch):
    """Advisories are additive. Losing them must not lose the costing."""

    monkeypatch.setattr(
        "backend.agents.regulatory.check_regulations",
        lambda code, description, language="en": (_ for _ in ()).throw(RuntimeError("429")),
    )

    results = [{"item": {"description": "polypropylene granules"},
                "chosen_hs_code": "3902.10.00",
                "duty": {"tti": 1234.0}}]
    out = builder.regulatory_node({"results": results})

    assert out["advisories"] == []
    assert results[0]["duty"]["tti"] == 1234.0


def test_verifier_failure_flags_for_review_rather_than_passing(monkeypatch):
    """The verifier exists to catch unsafe answers, so it failing is
    itself a reason to flag — never a quiet pass."""

    monkeypatch.setattr(
        "backend.agents.verifier.verify",
        lambda description, candidates, language="en": (_ for _ in ()).throw(RuntimeError("429")),
    )

    results = [{"item": {"description": "widget"}, "candidates": []}]
    out = builder.verify_node({"results": results})

    assert out["needs_review"] is True
    assert any("unconfirmed" in w.lower() for w in out["warnings"])