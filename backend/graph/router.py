"""
Router -- the node that keeps this from being a general chatbot.

Three outcomes, and the third one matters as much as the other two:

    invoice        a file was attached -> full pipeline
    duty_question  a tariff question   -> classify + calculate only
    out_of_scope   anything else       -> declined

Refusing is a feature. An agent that answers everything has no use case;
an agent that answers one thing well has a user. The refusal path is also
the cheapest thing to demonstrate on camera.

Routing is rule-first and LLM-second on purpose: a file attachment is a
fact, not a judgement, and spending a model call to notice it would be
latency for nothing.
"""

from __future__ import annotations

import re

from langsmith import traceable
from pydantic import BaseModel, Field

IN_SCOPE = re.compile(
    r"\b(hs\s*code|hs\b|tariff|duty|duties|customs|import|vat|"
    r"cd\b|sd\b|ait|landed|clearance|c&f|cnf|invoice|shipment|"
    r"শুল্ক|কর|আমদানি|ভ্যাট)\b",
    re.I,
)

ROUTE_SYSTEM = """\
Classify the user's message for a Bangladesh import-duty assistant.

  duty_question  asks about tariff classification, duty, VAT, customs
                 procedure, import requirements, or landed cost
  out_of_scope   anything else -- general chat, coding, writing, other
                 countries' customs, unrelated questions

The assistant does one job. Route generously within that job and firmly
outside it."""


class Route(BaseModel):
    route: str = Field(description="'duty_question' or 'out_of_scope'")
    reason: str = ""


@traceable(name="00_route", run_type="chain")
def route(state: dict) -> dict:
    """Decide which path the graph takes."""
    if state.get("raw_file_path"):
        return {"route": "invoice"}

    question = (state.get("question") or "").strip()
    if not question:
        return {"route": "out_of_scope"}

    # Fast path: obvious domain vocabulary needs no model call.
    if IN_SCOPE.search(question):
        return {"route": "duty_question"}

    try:
        from backend.core.llm import structured
        decision = structured("judge", Route).invoke([
            ("system", ROUTE_SYSTEM), ("user", question),
        ])
        chosen = decision.route if decision.route in {
            "duty_question", "out_of_scope"} else "out_of_scope"
        return {"route": chosen}
    except Exception:
        # If the model is unavailable, fail closed: better to decline
        # than to run the full pipeline on an unrelated question.
        return {"route": "out_of_scope"}


def pick_branch(state: dict) -> str:
    """Conditional edge function for the graph."""
    return state.get("route", "out_of_scope")
