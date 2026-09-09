"""
Exact HS code -> TariffRates lookup.

This is a tool, not an agent. It reads the processed schedule and returns
rates, or raises. It never guesses: if a code is not in the schedule, the
caller must surface that to the user rather than invent a number.
"""

from __future__ import annotations

import csv
import functools
from pathlib import Path

from backend.core.duty import TariffRates

TARIFF_CSV = Path("data/processed/tariff.csv")


class HSCodeNotFound(LookupError):
    """Raised when a code is absent from the loaded schedule."""


@functools.lru_cache(maxsize=1)
def _load_table() -> dict[str, dict]:
    if not TARIFF_CSV.exists():
        raise FileNotFoundError(
            f"{TARIFF_CSV} not found. Run: python -m backend.rag.ingest rates <pdf>"
        )
    with TARIFF_CSV.open(encoding="utf-8") as f:
        return {row["hs_code"]: row for row in csv.DictReader(f)}


def lookup_rates(hs_code: str) -> TariffRates:
    """Return the rates for one HS code.

    Args:
        hs_code: 8-digit code, e.g. "3901.20.10"

    Raises:
        HSCodeNotFound: if the code is not in the schedule.
    """
    table = _load_table()
    row = table.get(hs_code.strip())
    if row is None:
        raise HSCodeNotFound(f"{hs_code} is not in the loaded tariff schedule")

    return TariffRates.from_percentages(
        cd=float(row.get("cd") or 0),
        rd=float(row.get("rd") or 0),
        sd=float(row.get("sd") or 0),
        vat=float(row.get("vat") or 15),
        ait=float(row.get("ait") or 5),
        at=float(row.get("at") or 5),
    )


def describe(hs_code: str) -> str:
    """Schedule description for a code, for showing next to the number."""
    row = _load_table().get(hs_code.strip())
    if row is None:
        raise HSCodeNotFound(f"{hs_code} is not in the loaded tariff schedule")
    return row["description"]


def loaded_chapters() -> list[str]:
    """Which chapters the current build actually covers.

    Show this in the UI and state it in the README — honest scope beats a
    system that silently fails outside its coverage.
    """
    return sorted({row["chapter"] for row in _load_table().values()})
