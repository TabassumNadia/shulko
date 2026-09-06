"""
Bangladesh import-stage duty calculation.

This is the deterministic core of Shulko. There is NO LLM in this file.
The LLM's only job elsewhere is to decide WHICH tariff rates apply;
the arithmetic happens here, in plain Python, and is unit tested.

Cascade (import stage):
    AV  = (FOB + Freight + Insurance) * (1 + landing charge)
    CD  = AV * cd_rate
    RD  = AV * rd_rate
    SD  = (AV + CD + RD) * sd_rate
    VAT = (AV + CD + RD + SD) * vat_rate
    AIT = AV * ait_rate
    AT  = (AV + CD + RD + SD) * at_rate      # same base as VAT
    TTI = CD + RD + SD + VAT + AIT + AT
"""

from dataclasses import dataclass, asdict, field

# --- Assumptions, documented in README ------------------------------------
# Sources differ on some of these. They live here as named constants
# (never magic numbers buried in the formula) so they are easy to audit.
LANDING_CHARGE = 0.01   # 1% added to CIF to reach Assessable Value
DEFAULT_VAT = 0.15
DEFAULT_AIT = 0.05
DEFAULT_AT = 0.05


@dataclass
class TariffRates:
    """Rates for one HS code, as decimals (25% -> 0.25)."""
    cd: float = 0.0
    rd: float = 0.0
    sd: float = 0.0
    vat: float = DEFAULT_VAT
    ait: float = DEFAULT_AIT
    at: float = DEFAULT_AT

    @classmethod
    def from_percentages(cls, **kwargs):
        """TariffRates.from_percentages(cd=25, rd=5) -> rates as decimals."""
        return cls(**{k: v / 100.0 for k, v in kwargs.items()})


@dataclass
class DutyBreakdown:
    av: float
    cd: float
    rd: float
    sd: float
    vat: float
    ait: float
    at: float
    tti: float
    landed_cost: float
    effective_rate: float
    currency: str = "BDT"

    def to_dict(self):
        return asdict(self)


def calculate_duty(
    fob: float,
    freight: float,
    insurance: float,
    rates: TariffRates,
    importer_type: str = "commercial",
) -> DutyBreakdown:
    """
    Compute the full import-stage tax cascade for one line item.

    Args:
        fob: goods value in BDT (already converted from invoice currency)
        freight: shipping cost in BDT
        insurance: insurance cost in BDT
        rates: TariffRates for the classified HS code
        importer_type: "commercial" pays Advance Tax (AT); "industrial" does not

    Returns:
        DutyBreakdown with every component separated, so the UI can show
        the user exactly where each taka went.

    Raises:
        ValueError: on negative inputs or an unknown importer_type.
    """
    if min(fob, freight, insurance) < 0:
        raise ValueError("fob, freight and insurance must not be negative")
    if importer_type not in ("commercial", "industrial"):
        raise ValueError(f"unknown importer_type: {importer_type!r}")

    cif = fob + freight + insurance
    av = round(cif * (1 + LANDING_CHARGE), 2)

    cd = av * rates.cd
    rd = av * rates.rd
    sd = (av + cd + rd) * rates.sd

    # VAT and AT share the same base.
    vat_base = av + cd + rd + sd
    vat = vat_base * rates.vat
    ait = av * rates.ait
    at = vat_base * rates.at if importer_type == "commercial" else 0.0

    tti = cd + rd + sd + vat + ait + at

    return DutyBreakdown(
        av=round(av, 2),
        cd=round(cd, 2),
        rd=round(rd, 2),
        sd=round(sd, 2),
        vat=round(vat, 2),
        ait=round(ait, 2),
        at=round(at, 2),
        tti=round(tti, 2),
        landed_cost=round(av + tti, 2),
        effective_rate=round(tti / av * 100, 2) if av else 0.0,
    )
