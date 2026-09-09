"""
Validation of the duty engine against NBR's own published figures.

The consolidated tariff schedule prints a Total Tax Incidence (TTI)
column for every line: the total tax as a percentage of assessable
value, computed by the revenue authority itself.

That gives us an independent oracle. If our cascade is right, running it
on an assessable value of 100 must reproduce the published TTI for every
row in the schedule — thousands of rows, none of which we chose.

This is the test to open on camera. It is the difference between
"I implemented a formula I read somewhere" and "I proved my formula
matches the government's own numbers."

Note on assessable value: TTI is published as a percentage of AV, so
these checks start from AV = 100 directly and skip the 1% landing
charge, which applies when converting CIF to AV, not afterwards.
"""

import csv
from pathlib import Path

import pytest

TARIFF_CSV = Path("data/processed/tariff.csv")
TOLERANCE = 0.06          # published figures are rounded to 2 decimals


def _rows():
    if not TARIFF_CSV.exists():
        pytest.skip("data/processed/tariff.csv not built yet")
    with TARIFF_CSV.open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("tti_published")]


def cascade_on_100(row: dict) -> float:
    """Run the published cascade on AV = 100 and return the total."""
    av = 100.0
    cd = av * float(row["cd"]) / 100
    rd = av * float(row["rd"]) / 100
    sd = (av + cd + rd) * float(row["sd"]) / 100
    base = av + cd + rd + sd                  # VAT and AT share this base
    vat = base * float(row["vat"]) / 100
    ait = av * float(row["ait"]) / 100
    at = base * float(row["at"]) / 100
    return cd + rd + sd + vat + ait + at


def test_schedule_is_loaded():
    rows = _rows()
    assert len(rows) > 1000, "the schedule looks truncated"


def test_cascade_reproduces_every_published_tti():
    """The headline check: our arithmetic against NBR's, row by row."""
    rows = _rows()
    mismatches = [
        (r["hs_code"], float(r["tti_published"]), round(cascade_on_100(r), 2))
        for r in rows
        if abs(cascade_on_100(r) - float(r["tti_published"])) > TOLERANCE
    ]
    assert not mismatches, (
        f"{len(mismatches)} of {len(rows)} rows disagree with the published "
        f"TTI. First five: {mismatches[:5]}"
    )


def _cascade_with_at_excluding_rd(row: dict) -> float:
    """The rival formula: several published guides put AT on AV+CD+SD."""
    av = 100.0
    cd = av * float(row["cd"]) / 100
    rd = av * float(row["rd"]) / 100
    sd = (av + cd + rd) * float(row["sd"]) / 100
    vat = (av + cd + rd + sd) * float(row["vat"]) / 100
    ait = av * float(row["ait"]) / 100
    at = (av + cd + sd) * float(row["at"]) / 100        # RD omitted
    return cd + rd + sd + vat + ait + at


def test_advance_tax_shares_the_vat_base_not_the_narrower_one():
    """Settles a disagreement between published guides, with evidence.

    Some sources put AT on AV+CD+SD, omitting RD. On rows that carry both
    a regulatory duty and an advance tax the two formulas diverge, so the
    published TTI decides between them. It picks ours.
    """
    rows = [r for r in _rows() if float(r["rd"]) > 0 and float(r["at"]) > 0]
    assert rows, "no rows exercise both RD and AT, so nothing is proved"

    sample = rows[:300]
    ours = sum(abs(cascade_on_100(r) - float(r["tti_published"])) <= TOLERANCE
               for r in sample)
    rival = sum(abs(_cascade_with_at_excluding_rd(r) - float(r["tti_published"])) <= TOLERANCE
                for r in sample)

    assert ours == len(sample), f"our formula missed {len(sample) - ours} rows"
    assert rival == 0, "the rival formula should miss every one of these rows"


def test_no_row_has_a_negative_or_absurd_rate():
    for r in _rows():
        for field in ("cd", "rd", "sd", "vat", "ait", "at"):
            value = float(r[field])
            assert 0 <= value <= 500, f"{r['hs_code']} has {field}={value}"


def test_hs_codes_are_well_formed():
    import re
    pattern = re.compile(r"^\d{4}\.\d{2}\.\d{2}$")
    for r in _rows()[:500]:
        assert pattern.match(r["hs_code"]), r["hs_code"]
