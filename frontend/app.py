"""Streamlit front-end. Runs standalone today; wired to the API at H+17."""

import streamlit as st

from backend.core.duty import TariffRates, calculate_duty
from backend.core.i18n import t

st.set_page_config(page_title="Shulko", page_icon="🛃", layout="wide")

lang = st.sidebar.radio("Language / ভাষা", ["en", "bn"],
                        format_func=lambda x: "English" if x == "en" else "বাংলা")

st.title(t("app_title", lang))

# --- Temporary calculator view, so there is something runnable from hour one.
# Replaced by the full upload -> review -> report flow in the H+17 block.
st.header(t("report_header", lang))

c1, c2, c3 = st.columns(3)
fob = c1.number_input("FOB (BDT)", min_value=0.0, value=1000.0, step=100.0)
freight = c2.number_input(t("freight", lang), min_value=0.0, value=100.0, step=50.0)
insurance = c3.number_input(t("insurance", lang), min_value=0.0, value=50.0, step=10.0)

d1, d2, d3 = st.columns(3)
cd = d1.number_input("CD %", min_value=0.0, max_value=100.0, value=25.0)
rd = d2.number_input("RD %", min_value=0.0, max_value=100.0, value=5.0)
sd = d3.number_input("SD %", min_value=0.0, max_value=100.0, value=20.0)

importer_type = st.selectbox(
    t("importer_type", lang), ["commercial", "industrial"],
    format_func=lambda x: t(x, lang),
)

if st.button(t("analyze", lang), type="primary"):
    rates = TariffRates.from_percentages(cd=cd, rd=rd, sd=sd, vat=15, ait=5, at=5)
    r = calculate_duty(fob, freight, insurance, rates, importer_type)

    m1, m2, m3 = st.columns(3)
    m1.metric(t("assessable_value", lang), f"৳{r.av:,.2f}")
    m2.metric(t("total_tax", lang), f"৳{r.tti:,.2f}", f"{r.effective_rate}%")
    m3.metric(t("landed_cost", lang), f"৳{r.landed_cost:,.2f}")

    st.table({
        "Component": ["CD", "RD", "SD", "VAT", "AIT", "AT"],
        "BDT": [r.cd, r.rd, r.sd, r.vat, r.ait, r.at],
    })
