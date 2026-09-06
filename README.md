# Shulko (শুল্ক) — Bangladesh Import Landed-Cost Agent

Turns a photographed commercial invoice into an auditable landed-cost sheet:
every taka traced back to a tariff heading and a customs note.

> **Status:** scaffold. See the build sheet for the hour-by-hour plan.

## The problem

An SME importer in Dhaka does not know the duty payable until the goods
reach the port. Correct classification requires an 8-digit HS code drawn
from a 1,000-page tariff schedule full of exclusion notes, and NBR issues
SROs that change rates after the book is printed.

## How it works

```
Upload -> Router -> Extractor (OCR) -> Classifier (RAG) -> Duty Calculator
       -> Regulatory Agent (grounded search) -> Verifier -> Report
```

The LLM never computes a number. It decides which tariff heading applies and
cites the note that justifies it; the arithmetic is deterministic Python,
unit tested against a worked example from the customs rules.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in your keys
pytest                           # the duty engine must be green
streamlit run frontend/app.py
```

## Environment variables

See `.env.example`. No key is ever hardcoded in source.

## Assumptions

- Assessable Value = CIF + 1% landing charge.
- VAT and Advance Tax (AT) share the base `AV + CD + RD + SD`.
- AT applies to commercial importers only.
- Rates are FY 2025-26. Sources differ on some bases, so every rate lives
  in `backend/core/duty.py` as a named constant rather than a magic number.

## Research basis

| Paper | What it contributed |
|---|---|
| arXiv 2605.14857 — A Deterministic Agentic Workflow for HS Tariff Classification | The fixed multi-stage classifier; top-3 output because 65% of errors are ranking, not retrieval |
| ACL 2026 — HSGraphAgent | Hierarchical Select/Redirect traversal instead of flat retrieval |
| arXiv 2605.06635 — Cited but Not Verified | The Verifier Agent; citation accuracy is only 39-77% without one |
| arXiv 2603.07379 — SoK: Agentic RAG | Why the graph is deterministic rather than a free-roaming ReAct agent |
| arXiv 2606.16987 — Consensus-based Agentic LLM for HTS | Confidence scores and human-in-the-loop review in the UI |

## Limitations

- Tariff coverage limited to selected chapters (see `data/processed/`).
- Classification resolves to 6 digits; 8-digit is future work.
- SRO detection is search-based, not an authoritative feed.
