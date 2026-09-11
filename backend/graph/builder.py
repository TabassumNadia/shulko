"""
The graph.

    route ──┬─ invoice ───→ extract → classify_and_cost → regulatory → verify → report
            ├─ duty_question ──────→ classify_and_cost ──────────────────────→ report
            └─ out_of_scope ───────────────────────────────────────────────→ report

Deliberately a fixed graph, not an autonomous agent loop.

The important separation is:

    classifier
        ↓
    candidate + reasoning + evidence
        ↓
    verifier
        ↓
    verification result / warnings
        ↓
    report

Verifier feedback is kept separate from classifier reasoning.
"""

from __future__ import annotations

import concurrent.futures
import contextvars

from langsmith import traceable

from backend.core.duty import calculate_duty
from backend.core.i18n import tr
from backend.core.schemas import ItemAttributes, LineItem, LineItemResult
from backend.graph.router import pick_branch, route
from backend.graph.state import ShulkoState
from backend.tools.tariff_tool import HSCodeNotFound, lookup_rates


def _parallel(fn, jobs: list, workers: int | None = None) -> list:
    """Run independent per-item work concurrently while preserving tracing."""

    if len(jobs) <= 1:
        return [fn(job) for job in jobs]

    if workers is None:
        from backend.core.config import get_settings
        workers = max(get_settings().max_workers, 1)

    results = [None] * len(jobs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}

        for i, job in enumerate(jobs):
            ctx = contextvars.copy_context()
            futures[pool.submit(ctx.run, fn, job)] = i

        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()

    return results


OUT_OF_SCOPE_REPLY = {
    "en": (
        "I only handle Bangladesh import duty and HS classification. "
        "Attach a commercial invoice, or ask about a tariff code."
    ),
    "bn": (
        "আমি শুধু বাংলাদেশের আমদানি শুল্ক আর HS classification নিয়ে কাজ করি। "
        "একটা commercial invoice দিন, অথবা কোনো tariff code নিয়ে জিজ্ঞেস করুন।"
    ),
}


# ------------------------------------------------------------------
# 01 · EXTRACTION
# ------------------------------------------------------------------

@traceable(name="01_extract", run_type="chain")
def extract_node(state: ShulkoState) -> dict:
    from backend.agents.extractor import extract

    extracted = extract(state["raw_file_path"])
    lang = state.get("language", "en")

    return {
        "line_items": [i.model_dump() for i in extracted.items],
        "warnings": (
            []
            if extracted.items
            else [tr("no_line_items", lang)]
        ),
    }


# ------------------------------------------------------------------
# 02 + 03 · CLASSIFICATION + COST
# ------------------------------------------------------------------

@traceable(name="02_03_classify_and_cost", run_type="chain")
def classify_and_cost_node(state: ShulkoState) -> dict:
    """
    Classify each item and calculate duty.

    The classifier supplies:
        - candidate HS codes
        - reasoning
        - evidence
        - confidence

    Python calculates the duty. No LLM-generated number is used for
    arithmetic.
    """

    from backend.agents.classifier import classify

    items = [
        LineItem(**i)
        for i in state.get("line_items", [])
    ]

    if not items and state.get("question"):
        items = [
            LineItem(
                description=state["question"],
                attributes=ItemAttributes(),
            )
        ]

    freight = float(state.get("freight") or 0)
    insurance = float(state.get("insurance") or 0)
    importer_type = state.get("importer_type") or "commercial"
    lang = state.get("language", "en")

    # A duty question has no real invoice value.
    # We therefore show statutory rates rather than pretending that
    # ৳0.00 is the user's actual duty.
    notional = all(
        i.quantity * i.unit_price == 0
        for i in items
    )

    if notional:
        freight = insurance = 0.0

    # Shipment-level freight and insurance are allocated by item value.
    total_value = (
        sum(i.quantity * i.unit_price for i in items)
        or 1.0
    )

    def process(item: LineItem) -> dict:
        result = LineItemResult(item=item)

        try:
            candidates = classify(item, lang)
        except Exception as exc:
            result.needs_review = True
            result.warnings.append(
                tr(
                    "classification_failed",
                    lang,
                    error=exc,
                )
            )
            return result.model_dump()

        result.candidates = candidates

        if not candidates:
            result.needs_review = True
            result.warnings.append(
                tr("no_heading_matched", lang)
            )
            return result.model_dump()

        chosen = candidates[0]
        result.chosen_hs_code = chosen.hs_code

        share = (
            (item.quantity * item.unit_price)
            / total_value
        )

        try:
            rates = lookup_rates(chosen.hs_code)

            duty = calculate_duty(
                fob=(
                    100.0
                    if notional
                    else item.quantity * item.unit_price
                ),
                freight=freight * share,
                insurance=insurance * share,
                rates=rates,
                importer_type=importer_type,
            )

            result.duty = duty.to_dict()

            result.rates_pct = {
                "cd": round(rates.cd * 100, 2),
                "rd": round(rates.rd * 100, 2),
                "sd": round(rates.sd * 100, 2),
                "vat": round(rates.vat * 100, 2),
                "ait": round(rates.ait * 100, 2),
                "at": round(rates.at * 100, 2),
                "total_tax_incidence": duty.effective_rate,
            }

        except HSCodeNotFound:
            result.needs_review = True
            result.warnings.append(
                tr(
                    "hs_code_not_found",
                    lang,
                    code=chosen.hs_code,
                )
            )

        return result.model_dump()

    results = _parallel(process, items)

    return {
        "results": results,
        "notional": notional,
    }


# ------------------------------------------------------------------
# 04 · REGULATORY
# ------------------------------------------------------------------

@traceable(name="04_regulatory", run_type="chain")
def regulatory_node(state: ShulkoState) -> dict:
    """
    Run live regulatory checks once per distinct HS code.

    Regulatory advisories remain separate from classification reasoning
    and verification warnings.
    """

    from backend.agents.regulatory import check_regulations
    from backend.core.config import get_settings

    results = state.get("results", [])
    limit = get_settings().max_regulatory_checks

    unique: dict[str, str] = {}

    for r in results:
        code = r.get("chosen_hs_code")

        if code and code not in unique:
            unique[code] = r["item"]["description"]

        if len(unique) >= limit:
            break

    if not unique:
        return {
            "advisories": [],
            "grounding": {
                "mechanism": "none",
                "available": False,
                "codes_checked": 0,
            },
        }

    def check(pair):
        code, description = pair

        try:
            result = check_regulations(
                code,
                description,
                state.get("language", "en"),
            )

            return (
                code,
                [a.model_dump() for a in result.advisories],
                result.mechanism,
            )

        except Exception as exc:
            print(
                f"[regulatory] check failed for {code}: {exc}"
            )
            return code, [], "none"

    pairs = list(unique.items())
    found = _parallel(check, pairs)

    by_code = {
        code: items
        for code, items, _ in found
    }

    advisories = []

    for r in results:
        code = r.get("chosen_hs_code")

        r["advisories"] = by_code.get(code, [])
        advisories.extend(r["advisories"])

    ran = {
        mechanism
        for _, _, mechanism in found
    }

    mechanism = next(
        (
            m
            for m in ("google_search", "tavily")
            if m in ran
        ),
        "none",
    )

    return {
        "advisories": advisories,
        "grounding": {
            "mechanism": mechanism,
            "available": mechanism != "none",
            "codes_checked": len(pairs),
        },
    }


# ------------------------------------------------------------------
# 05 · VERIFICATION
# ------------------------------------------------------------------

@traceable(name="05_verify", run_type="chain")
def verify_node(state: ShulkoState) -> dict:
    """
    Verify classifier output.

    IMPORTANT:
    Verifier warnings are stored separately from classifier reasoning.

    result["warnings"]
        = extraction/classification/calculation problems

    result["verification"]
        = verifier verdict and verification-specific warnings

    This prevents technical verifier feedback from appearing inside
    the classifier's "Why this HS code?" explanation.
    """

    from backend.agents.classifier import HSCandidate
    from backend.agents.verifier import Verdict, verify

    results = state.get("results", [])
    lang = state.get("language", "en")

    def judge(result: dict):
        candidates = [
            HSCandidate(**c)
            for c in result.get("candidates", [])
        ]

        try:
            return verify(
                result["item"]["description"],
                candidates,
                lang,
            )

        except Exception as exc:
            return Verdict(
                verdict="fail",
                needs_review=True,
                warnings=[
                    tr(
                        "verification_failed",
                        lang,
                        error=exc,
                    )
                ],
            )

    verdicts = _parallel(judge, results)

    needs_review = False
    workflow_warnings: list[str] = []

    for result, verdict in zip(results, verdicts):

        # Keep the classifier warnings exactly where they belong.
        existing_warnings = result.get("warnings", [])

        # Store verifier information separately.
        verification = {
            "verdict": verdict.verdict,
            "code_matches": verdict.code_matches,
            "evidence_supported": verdict.evidence_supported,
            "confidence_justified": verdict.confidence_justified,
            "needs_review": verdict.needs_review,
            "warnings": list(verdict.warnings or []),
        }

        result["verification"] = verification

        # A failed verification means the classification must not be
        # treated as filing-ready.
        if verdict.needs_review or verdict.verdict != "pass":
            result["needs_review"] = True
            needs_review = True

        # Do NOT merge verifier warnings into result["warnings"].
        #
        # That was the reason the UI previously displayed a long block
        # of technical messages directly under the item.
        result["warnings"] = existing_warnings

        # Keep workflow-level warnings separately.
        for warning in verdict.warnings or []:
            workflow_warnings.append(warning)

    return {
        "needs_review": needs_review,
        "warnings": workflow_warnings,
    }


# ------------------------------------------------------------------
# 06 · REPORT
# ------------------------------------------------------------------

@traceable(name="06_report", run_type="chain")
def report_node(state: ShulkoState) -> dict:
    lang = state.get("language", "en")

    if state.get("route") == "out_of_scope":
        return {
            "report": {
                "kind": "refusal",
                "message": OUT_OF_SCOPE_REPLY.get(
                    lang,
                    OUT_OF_SCOPE_REPLY["en"],
                ),
            }
        }

    results = state.get("results", [])

    totals = {
        "av": 0.0,
        "tti": 0.0,
        "landed_cost": 0.0,
    }

    for r in results:
        duty = r.get("duty") or {}

        for key in totals:
            totals[key] += float(
                duty.get(key, 0.0)
            )

    return {
        "report": {
            "kind": (
                "rate_card"
                if state.get("notional")
                else "landed_cost"
            ),

            "items": results,

            "totals": {
                k: round(v, 2)
                for k, v in totals.items()
            },

            "effective_rate": (
                round(
                    totals["tti"]
                    / totals["av"]
                    * 100,
                    2,
                )
                if totals["av"]
                else 0.0
            ),

            "needs_review": state.get(
                "needs_review",
                False,
            ),

            "warnings": state.get(
                "warnings",
                [],
            ),

            "advisories": state.get(
                "advisories",
                [],
            ),

            "grounding": state.get(
                "grounding"
            )
            or {
                "mechanism": "none",
                "available": False,
                "codes_checked": 0,
            },
        }
    }


# ------------------------------------------------------------------
# GRAPH WIRING
# ------------------------------------------------------------------

def build_graph():
    """Compile the fixed Shulko workflow."""

    from langgraph.graph import END, StateGraph

    g = StateGraph(ShulkoState)

    g.add_node("route", route)
    g.add_node("extract", extract_node)
    g.add_node(
        "classify_and_cost",
        classify_and_cost_node,
    )
    g.add_node("regulatory", regulatory_node)
    g.add_node("verify", verify_node)
    g.add_node("report", report_node)

    g.set_entry_point("route")

    g.add_conditional_edges(
        "route",
        pick_branch,
        {
            "invoice": "extract",
            "duty_question": "classify_and_cost",
            "out_of_scope": "report",
        },
    )

    g.add_edge(
        "extract",
        "classify_and_cost",
    )

    g.add_edge(
        "classify_and_cost",
        "regulatory",
    )

    g.add_edge(
        "regulatory",
        "verify",
    )

    g.add_edge(
        "verify",
        "report",
    )

    g.add_edge(
        "report",
        END,
    )

    return g.compile()


_graph = None


def get_graph():
    global _graph

    if _graph is None:
        _graph = build_graph()

    return _graph