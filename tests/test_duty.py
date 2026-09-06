"""
Unit tests for the duty engine.

The headline test is the worked example from the Bangladesh customs
calculation guide. If this passes, the arithmetic core is trustworthy —
and you can say so, on camera, with a straight face.
"""

import pytest

from backend.core.duty import TariffRates, calculate_duty, LANDING_CHARGE


def test_worked_example_from_customs_guide():
    """
    FOB 1000, Freight 100, Insurance 50
    CD 25% | RD 5% | SD 20% | VAT 15% | AIT 5% | AT 5%
    """
    rates = TariffRates.from_percentages(cd=25, rd=5, sd=20, vat=15, ait=5, at=5)
    r = calculate_duty(fob=1000, freight=100, insurance=50, rates=rates)

    assert r.av == 1161.50
    assert r.cd == 290.38
    assert r.rd == 58.08
    assert r.sd == 301.99
    assert r.vat == 271.79
    assert r.ait == 58.08
    assert r.at == 90.60
    assert r.tti == 1070.90


def test_landed_cost_is_av_plus_tti():
    rates = TariffRates.from_percentages(cd=25, rd=5, sd=20, vat=15, ait=5, at=5)
    r = calculate_duty(fob=1000, freight=100, insurance=50, rates=rates)
    assert r.landed_cost == pytest.approx(r.av + r.tti, abs=0.01)


def test_industrial_importer_pays_no_advance_tax():
    rates = TariffRates.from_percentages(cd=25, rd=5, sd=20, vat=15, ait=5, at=5)
    commercial = calculate_duty(1000, 100, 50, rates, importer_type="commercial")
    industrial = calculate_duty(1000, 100, 50, rates, importer_type="industrial")

    assert industrial.at == 0.0
    assert industrial.tti < commercial.tti


def test_zero_rated_goods_still_pay_vat():
    """A 0% CD item is not tax free — VAT still applies on the AV."""
    rates = TariffRates.from_percentages(cd=0, rd=0, sd=0, vat=15, ait=5, at=5)
    r = calculate_duty(fob=1000, freight=0, insurance=0, rates=rates)

    assert r.cd == 0.0
    assert r.vat > 0
    assert r.tti > 0


def test_assessable_value_includes_landing_charge():
    rates = TariffRates()
    r = calculate_duty(fob=1000, freight=0, insurance=0, rates=rates)
    assert r.av == pytest.approx(1000 * (1 + LANDING_CHARGE), abs=0.01)


def test_negative_input_is_rejected():
    with pytest.raises(ValueError):
        calculate_duty(fob=-1, freight=0, insurance=0, rates=TariffRates())


def test_unknown_importer_type_is_rejected():
    with pytest.raises(ValueError):
        calculate_duty(1000, 0, 0, TariffRates(), importer_type="hobbyist")
