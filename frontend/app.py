"""
Shulko — Streamlit front-end.

A conversation on the outside, a fixed workflow underneath. Every message
takes one of three paths, decided by the Router:

    attach an invoice   ->  OCR -> classify -> cost -> ground -> verify
    ask about duty      ->  classify -> cost
    anything else       ->  declined, in scope

The refusal path matters as much as the other two. An assistant that
answers everything has no use case; one that answers a single question
well has a user.

This file holds no business logic. It calls the API and renders what
comes back — so the backend stays the thing worth explaining, and the
UI can be swapped without touching a single agent.
"""

from __future__ import annotations

import io

import pandas as pd
import requests
import streamlit as st

API = "http://localhost:8000"
# The invoice path runs OCR, then classification, grounded search and
# verification for every line item. Those run concurrently, but a slow
# grounded search can still push one run past three minutes, and a
# timeout mid-demo looks like a crash.
TIMEOUT = 420

st.set_page_config(page_title="Shulko", page_icon="🛃", layout="centered")


# ---------------------------------------------------------------- strings
TEXT = {
    "en": {
        "rate_card": "Statutory rates for this heading",
        "rate_note": "You asked about a code, not a shipment — so these are "
                     "the rates, not an amount. Attach an invoice for taka figures.",
        "tti": "Total tax incidence",
        "rate": "Rate",
        "title": "Shulko",
        "tagline": "Bangladesh import landed-cost agent",
        "greeting": (
            "Attach a commercial invoice in the sidebar, or ask me about duty "
            "on an HS code.\n\nTry: *what is the import duty on polypropylene "
            "granules?*"
        ),
        "shipment": "Shipment costs",
        "shipment_help": "Your invoice prices the goods only (FOB). Bangladesh "
                         "customs charges duty on the landed value, so freight "
                         "and insurance are added before any rate applies. "
                         "Enter your own figures — these are just starting values.",
        "freight": "Freight (BDT)",
        "freight_help": "What you pay to ship the goods to Chattogram. Any amount.",
        "insurance": "Insurance (BDT)",
        "insurance_help": "Marine insurance on the shipment. Enter 0 if uninsured.",
        "importer": "Importer type",
        "importer_help": "Commercial importers pay Advance Tax (AT); "
                         "industrial importers do not.",
        "analyze": "Analyze invoice",
        "note": "Anything to add? (optional)",
        "note_ph": "e.g. these are for a VAT-registered cable factory",
        "ready": "Ready — press Analyze invoice",
        "commercial": "Commercial",
        "industrial": "Industrial",
        "invoice": "Commercial invoice",
        "clear": "Clear conversation",
        "ask": "Ask about a tariff code…",
        "av": "Assessable value",
        "tax": "Total tax",
        "landed": "Landed cost",
        "why": "Why this HS code?",
        "alternatives": "Other candidates the classifier considered",
        "review": "Needs human review before filing",
        "advisories": "Regulatory advisories",
        "download": "Download as CSV",
        "thinking": "Running the pipeline…",
        "offline": "Backend not reachable. Start it with: uvicorn backend.main:app --reload",
        "confidence": "confidence",
        "no_items": "No line items could be read from that document.",
        "grounding_via": "Regulatory advisories above were retrieved live via {engine}.",
        "grounding_clear": "Checked live via {engine}: no additional Bangladesh "
                           "import restriction or certificate requirement was found "
                           "for these codes.",
        "grounding_off": "The live regulatory check could not run (no search key, "
                         "or the daily API quota is exhausted), so this run carries "
                         "no advisories. Absence here does not mean none apply — "
                         "verify SRO and certificate requirements separately.",
    },
    "bn": {
        "rate_card": "এই heading-এর সরকারি হার",
        "rate_note": "আপনি একটা code নিয়ে জিজ্ঞেস করেছেন, চালান নিয়ে নয় — তাই এখানে "
                     "হার দেখানো হচ্ছে, টাকার অঙ্ক নয়। টাকার হিসাব পেতে invoice দিন।",
        "tti": "মোট করভার",
        "rate": "হার",
        "title": "শুল্ক",
        "tagline": "বাংলাদেশের আমদানি খরচ নির্ণয়ক",
        "greeting": (
            "পাশের প্যানেলে commercial invoice দিন, অথবা কোনো HS code-এর শুল্ক "
            "জিজ্ঞেস করুন।\n\nযেমন: *পলিপ্রোপিলিন দানার আমদানি শুল্ক কত?*"
        ),
        "shipment": "চালানের খরচ",
        "shipment_help": "Invoice-এ শুধু পণ্যের দাম থাকে (FOB)। কিন্তু কাস্টমস শুল্ক "
                         "বসায় বন্দরে পৌঁছানো মোট মূল্যের উপর — তাই জাহাজ ভাড়া আর "
                         "বীমা আগে যোগ হয়। আপনার নিজের অঙ্ক বসান; এগুলো শুধু শুরুর মান।",
        "freight": "জাহাজ ভাড়া (টাকা)",
        "freight_help": "চট্টগ্রাম পর্যন্ত মাল আনতে যা খরচ। যেকোনো অঙ্ক দিতে পারেন।",
        "insurance": "বীমা (টাকা)",
        "insurance_help": "চালানের সামুদ্রিক বীমা। বীমা না থাকলে ০ দিন।",
        "importer": "আমদানিকারকের ধরন",
        "importer_help": "বাণিজ্যিক আমদানিকারক Advance Tax (AT) দেয়, শিল্প দেয় না।",
        "analyze": "হিসাব করুন",
        "note": "কিছু যোগ করতে চান? (ঐচ্ছিক)",
        "note_ph": "যেমন: এগুলো VAT-নিবন্ধিত ক্যাবল কারখানার জন্য",
        "ready": "প্রস্তুত — \"হিসাব করুন\" চাপুন",
        "commercial": "বাণিজ্যিক",
        "industrial": "শিল্প",
        "invoice": "কমার্শিয়াল ইনভয়েস",
        "clear": "কথোপকথন মুছুন",
        "ask": "কোনো tariff code নিয়ে জিজ্ঞেস করুন…",
        "av": "নিরূপিত মূল্য",
        "tax": "মোট কর",
        "landed": "সর্বমোট খরচ",
        "why": "এই HS code কেন?",
        "alternatives": "বিবেচনায় আসা অন্যান্য code",
        "review": "দাখিলের আগে যাচাই প্রয়োজন",
        "advisories": "নিয়মকানুন সংক্রান্ত সতর্কতা",
        "download": "CSV ডাউনলোড",
        "thinking": "প্রক্রিয়া চলছে…",
        "offline": "ব্যাকএন্ড চলছে না। চালু করুন: uvicorn backend.main:app --reload",
        "confidence": "নিশ্চয়তা",
        "no_items": "এই ডকুমেন্ট থেকে কোনো পণ্য পড়া যায়নি।",
        "grounding_via": "উপরের নিয়মকানুন সংক্রান্ত তথ্য {engine} দিয়ে সরাসরি "
                         "ইন্টারনেট থেকে আনা হয়েছে।",
        "grounding_clear": "{engine} দিয়ে যাচাই করা হয়েছে — এই code-গুলোর জন্য "
                           "অতিরিক্ত কোনো আমদানি নিষেধাজ্ঞা বা সার্টিফিকেটের "
                           "প্রয়োজন পাওয়া যায়নি।",
        "grounding_off": "সরাসরি নিয়মকানুন যাচাই করা যায়নি (search key নেই, "
                         "অথবা দৈনিক API কোটা শেষ), তাই এই ফলাফলে কোনো সতর্কতা "
                         "নেই। এর মানে কোনো নিয়ম নেই তা নয় — SRO আর সার্টিফিকেটের "
                         "শর্ত আলাদাভাবে যাচাই করুন।",
    },
}


def t(key: str, lang: str) -> str:
    return TEXT.get(lang, TEXT["en"]).get(key, TEXT["en"].get(key, key))


# ---------------------------------------------------------------- helpers
@st.cache_data(ttl=15)
def health() -> dict | None:
    try:
        return requests.get(f"{API}/health", timeout=5).json()
    except Exception:
        return None


def taka(value: float) -> str:
    return f"৳{value:,.2f}"


def _render_grounding(report: dict, lang: str) -> None:
    """Say whether the live regulatory search actually ran.

    An empty advisory list is ambiguous on its own: it can mean the web
    was searched and Bangladesh imposes no extra obligation on this code,
    or that no search happened at all. Only the first is reassuring, so
    the two must never render identically.
    """
    grounding = report.get("grounding") or {}
    if not grounding.get("codes_checked"):
        return

    if not grounding.get("available"):
        st.caption(t("grounding_off", lang))
        return

    engine = ("Google Search grounding"
              if grounding.get("mechanism") == "google_search"
              else "Tavily search (fallback)")
    key = "grounding_via" if (report.get("advisories") or []) else "grounding_clear"
    st.caption(t(key, lang).format(engine=engine))


def render_report(report: dict, lang: str) -> None:
    """Render one API response inside a chat message."""
    if not report:
        return

    if report.get("kind") == "refusal":
        st.info(report.get("message", ""))
        return

    items = report.get("items") or []
    if not items:
        st.warning(t("no_items", lang))
        return

    # A typed question has no invoice value behind it. Showing ৳0.00 would
    # read as an answer rather than an absence, so that path renders the
    # statutory rate card instead.
    if report.get("kind") == "rate_card":
        st.caption(t("rate_note", lang))
        for result in items:
            _render_rate_card(result, lang)
        _render_grounding(report, lang)
        return

    totals = report.get("totals") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric(t("av", lang), taka(totals.get("av", 0)))
    c2.metric(t("tax", lang), taka(totals.get("tti", 0)),
              f"{report.get('effective_rate', 0)}%")
    c3.metric(t("landed", lang), taka(totals.get("landed_cost", 0)))

    if report.get("needs_review"):
        st.warning(f"⚠ {t('review', lang)}")

    for result in items:
        _render_item(result, lang)

    advisories = report.get("advisories") or []
    if advisories:
        st.markdown(f"**{t('advisories', lang)}**")
        for a in advisories:
            st.markdown(f"- {a['message']}  \n  [{a.get('kind', 'info')}]({a['source_url']})")

    _render_grounding(report, lang)

    st.download_button(
        t("download", lang),
        data=_to_csv(items),
        file_name="shulko-landed-cost.csv",
        mime="text/csv",
        key=f"dl-{id(report)}",
    )


def _render_rate_card(result: dict, lang: str) -> None:
    """Rates as percentages, for a question with no shipment attached."""
    item = result.get("item", {})
    candidates = result.get("candidates") or []
    rates = result.get("rates_pct") or {}
    top = candidates[0] if candidates else None

    header = item.get("description", "item")
    if result.get("chosen_hs_code"):
        header = f"{result['chosen_hs_code']} · {header}"

    with st.container(border=True):
        st.markdown(f"**{header}**")
        if top:
            pct = int(top.get("confidence", 0) * 100)
            st.caption(f"{top.get('heading_text', '')} — {pct}% {t('confidence', lang)}")
            st.progress(min(max(top.get("confidence", 0.0), 0.0), 1.0))

        if rates:
            st.metric(t("tti", lang), f"{rates.get('total_tax_incidence', 0)}%")
            st.dataframe(
                pd.DataFrame({
                    "Component": ["CD", "RD", "SD", "VAT", "AIT", "AT"],
                    t("rate", lang): [f"{rates.get(k, 0)}%" for k in
                                      ("cd", "rd", "sd", "vat", "ait", "at")],
                }),
                hide_index=True, width="stretch",
            )

        for warning in result.get("warnings") or []:
            st.warning(warning)

        if top:
            with st.expander(t("why", lang)):
                st.markdown(top.get("reasoning", ""))
                for e in top.get("evidence") or []:
                    mark = "✓" if e.get("supports", True) else "✗"
                    st.markdown(f"> {mark} **{e['source']}** — “{e['quote']}”")
                if len(candidates) > 1:
                    st.markdown(f"**{t('alternatives', lang)}**")
                    for c in candidates[1:]:
                        st.markdown(
                            f"- `{c['hs_code']}` "
                            f"({int(c.get('confidence', 0) * 100)}%) "
                            f"{c.get('heading_text', '')}"
                        )


def _render_item(result: dict, lang: str) -> None:
    item = result.get("item", {})
    candidates = result.get("candidates") or []
    duty = result.get("duty") or {}
    top = candidates[0] if candidates else None

    header = item.get("description", "item")
    if result.get("chosen_hs_code"):
        header = f"{result['chosen_hs_code']} · {header}"

    with st.container(border=True):
        st.markdown(f"**{header}**")

        if top:
            pct = int(top.get("confidence", 0) * 100)
            st.caption(f"{top.get('heading_text', '')} — {pct}% {t('confidence', lang)}")
            st.progress(min(max(top.get("confidence", 0.0), 0.0), 1.0))

        if duty:
            st.dataframe(
                pd.DataFrame({
                    "Component": ["CD", "RD", "SD", "VAT", "AIT", "AT", "Total"],
                    "BDT": [duty.get(k, 0) for k in
                            ("cd", "rd", "sd", "vat", "ait", "at", "tti")],
                }),
                hide_index=True, width="stretch",
            )

        for warning in result.get("warnings") or []:
            st.warning(warning)

        if top:
            with st.expander(t("why", lang)):
                st.markdown(top.get("reasoning", ""))
                for e in top.get("evidence") or []:
                    mark = "✓" if e.get("supports", True) else "✗"
                    st.markdown(f"> {mark} **{e['source']}** — “{e['quote']}”")

                if len(candidates) > 1:
                    st.markdown(f"**{t('alternatives', lang)}**")
                    for c in candidates[1:]:
                        st.markdown(
                            f"- `{c['hs_code']}` "
                            f"({int(c.get('confidence', 0) * 100)}%) "
                            f"{c.get('heading_text', '')}"
                        )


def _to_csv(items: list[dict]) -> bytes:
    rows = []
    for r in items:
        duty = r.get("duty") or {}
        rows.append({
            "description": r.get("item", {}).get("description", ""),
            "hs_code": r.get("chosen_hs_code", ""),
            "confidence": (r.get("candidates") or [{}])[0].get("confidence", 0),
            **{k: duty.get(k, 0) for k in
               ("av", "cd", "rd", "sd", "vat", "ait", "at", "tti", "landed_cost")},
            "needs_review": r.get("needs_review", False),
        })
    buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🛃 Shulko / শুল্ক")

    lang = st.radio(
        "Language / ভাষা", ["en", "bn"], horizontal=True,
        format_func=lambda x: "English" if x == "en" else "বাংলা",
    )

    status = health()
    if status is None:
        st.error(t("offline", lang))
    else:
        indexed = status.get("indexed", {})
        st.caption(
            f"{status.get('retrieval_mode', '?')} · "
            f"{indexed.get('tariff_lines', 0):,} tariff lines · "
            f"{indexed.get('chapter_notes', 0)} notes"
        )
        if not status.get("langsmith_tracing"):
            st.caption("⚠ LangSmith tracing off")

    st.divider()
    st.markdown(f"**{t('shipment', lang)}**")
    st.caption(t("shipment_help", lang))
    freight = st.number_input(t("freight", lang), min_value=0.0, value=5000.0,
                              step=500.0, help=t("freight_help", lang))
    insurance = st.number_input(t("insurance", lang), min_value=0.0, value=1000.0,
                                step=100.0, help=t("insurance_help", lang))
    importer_type = st.selectbox(
        t("importer", lang), ["commercial", "industrial"],
        format_func=lambda x: t(x, lang), help=t("importer_help", lang),
    )

    st.divider()
    invoice = st.file_uploader(
        t("invoice", lang), type=["png", "jpg", "jpeg", "webp", "pdf"]
    )

    # Attaching a file must not start the run: the user may still want to
    # add a note, change the freight, or simply look at what they picked.
    note = ""
    analyze = False
    if invoice is not None:
        note = st.text_input(t("note", lang), placeholder=t("note_ph", lang))
        analyze = st.button(t("analyze", lang), type="primary",
                            width="stretch")
        st.caption(t("ready", lang))

    if st.button(t("clear", lang), width="stretch"):
        st.session_state.messages = []
        st.session_state.handled = None
        st.rerun()


# ---------------------------------------------------------------- state
st.session_state.setdefault("messages", [])
st.session_state.setdefault("handled", None)


# ---------------------------------------------------------------- history
st.title(t("title", lang))
st.caption(t("tagline", lang))

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(t("greeting", lang))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("text"):
            st.markdown(message["text"])
        if message.get("report"):
            render_report(message["report"], lang)


# ---------------------------------------------------------------- invoice
if analyze and invoice is not None:
    st.session_state.handled = invoice.name
    label = f"📎 **{invoice.name}**"
    if note:
        label += f"\n\n{note}"
    st.session_state.messages.append({"role": "user", "text": label})
    with st.chat_message("user"):
        st.markdown(label)

    with st.chat_message("assistant"):
        with st.status(t("thinking", lang), expanded=True) as status_box:
            st.write("01 · OCR and line-item extraction")
            st.write("02 · HS classification against the tariff and chapter notes")
            st.write("03 · Duty calculation")
            st.write("04 · Live regulatory check")
            st.write("05 · Verification")
            try:
                response = requests.post(
                    f"{API}/analyze",
                    files={"file": (invoice.name, invoice.getvalue())},
                    data={
                        "freight": freight, "insurance": insurance,
                        "importer_type": importer_type, "language": lang,
                    },
                    timeout=TIMEOUT,
                )
                response.raise_for_status()
                payload = response.json()
                status_box.update(label="Done", state="complete", expanded=False)
            except Exception as exc:
                status_box.update(label="Failed", state="error")
                st.error(f"{exc}")
                payload = None

        if payload:
            report = payload.get("report")
            render_report(report, lang)
            st.session_state.messages.append(
                {"role": "assistant", "report": report}
            )


# ---------------------------------------------------------------- question
if prompt := st.chat_input(t("ask", lang)):
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(t("thinking", lang)):
            try:
                response = requests.post(
                    f"{API}/ask",
                    json={
                        "question": prompt, "language": lang,
                        "freight": freight, "insurance": insurance,
                        "importer_type": importer_type,
                    },
                    timeout=TIMEOUT,
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                st.error(f"{exc}")
                payload = None

        if payload:
            report = payload.get("report")
            render_report(report, lang)
            st.session_state.messages.append(
                {"role": "assistant", "report": report}
            )
