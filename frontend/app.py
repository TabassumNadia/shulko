"""
Shulko — Streamlit front-end.

The UI only renders API results.

Backend workflow:

    invoice
      ↓
    OCR
      ↓
    classification
      ↓
    duty calculation
      ↓
    regulatory check
      ↓
    verification
      ↓
    report

The UI deliberately keeps these concepts separate:
    - classifier reasoning
    - classifier evidence
    - verifier result
    - regulatory advisories
"""

from __future__ import annotations

import contextlib
import io
import re

import pandas as pd
import requests
import streamlit as st
from voice import build_plain_summary, speak

API = "http://localhost:8000"

TIMEOUT = 420


st.set_page_config(
    page_title="Shulko",
    page_icon="🛃",
    layout="centered",
)


# ------------------------------------------------------------------
# THEME -- customs/tax palette: navy + gold (seal) + teal (trust/money).
# Deliberately not the generic purple "AI orb" look -- this is a duty
# agent, so the accent colours read as official/financial instead.
# ------------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(
            circle at 15% 0%, #1b1a3a 0%, #0B0F1E 55%
        ) !important;
    }

    /* Sidebar forced permanently open -- it kept collapsing (Streamlit
       remembers the collapsed state per browser), and the button to
       reopen it was too easy to miss. Simplest fix: it can no longer
       collapse at all. */
    [data-testid="stSidebar"] {
        background: #10101f !important;
        border-right: 1px solid rgba(108, 99, 255, 0.18);
        min-width: 21rem !important;
        max-width: 21rem !important;
        transform: none !important;
        visibility: visible !important;
        margin-left: 0px !important;
    }
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    /* Default Streamlit chat avatars (a generic red/orange robot icon per
       role) -- not ours, and not needed; the message bubble is enough. */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"],
    [data-testid="stChatMessageAvatarCustom"] {
        display: none !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        text-align: left !important;
        background: #161532 !important;
        border: 1px solid rgba(108, 99, 255, 0.18) !important;
        border-radius: 10px !important;
    }

    [data-testid="stStatusWidget"] { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }

    /* The sidebar's own open/close arrow lives in Streamlit's header bar --
       it was invisible against the dark background, not actually gone. */
    header[data-testid="stHeader"] {
        background: transparent !important;
        width: 100% !important;
    }
    header[data-testid="stHeader"] * {
        color: #E7E8FF !important;
        fill: #E7E8FF !important;
    }

    [data-testid="stChatInput"] { border-radius: 999px !important; }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5, #6C63FF) !important;
        border: none !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        box-shadow: 0 0 18px rgba(108, 99, 255, 0.4);
    }

    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: rgba(108, 99, 255, 0.22) !important;
        color: #E7E8FF !important;
        border: 1px solid rgba(108, 99, 255, 0.5) !important;
        box-shadow: none !important;
    }

    [data-testid="stMetric"] {
        background: #14142c;
        border: 1px solid rgba(108, 99, 255, 0.22);
        border-radius: 12px;
        padding: 10px 14px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        border-color: rgba(108, 99, 255, 0.2) !important;
    }

    /* -- feature info cards, under the empty-state chat input -- */
    .shulko-feature {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 12px 14px;
        border-radius: 12px;
        background: #14142c;
        border: 1px solid rgba(108, 99, 255, 0.2);
    }
    .shulko-feature-icon {
        width: 34px;
        height: 34px;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 17px;
        border-radius: 9px;
        background: rgba(108, 99, 255, 0.18);
        animation: shulko-float 3s ease-in-out infinite;
    }
    .shulko-feature-title {
        font-size: 13.5px;
        font-weight: 600;
        color: #E7E8FF;
    }
    .shulko-feature-sub {
        font-size: 11.5px;
        color: #9AA0C0;
    }

    /* -- top-right online/offline status pill -- */
    .shulko-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 999px;
        background: #14142c;
        border: 1px solid rgba(108, 99, 255, 0.25);
        font-size: 12.5px;
        color: #C7C9F5;
    }
    .shulko-status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22C55E;
        box-shadow: 0 0 8px #22C55E;
        animation: shulko-pulse 1.6s ease-in-out infinite;
    }
    .shulko-status-dot.off {
        background: #EF4444;
        box-shadow: 0 0 8px #EF4444;
        animation: none;
    }

    /* -- login-page hero card -- */
    .shulko-hero {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 20px 24px;
        margin-bottom: 18px;
        border-radius: 16px;
        background: linear-gradient(
            135deg, rgba(99, 102, 241, 0.18), rgba(56, 189, 248, 0.12)
        );
        border: 1px solid rgba(108, 99, 255, 0.4);
        box-shadow: 0 0 40px rgba(108, 99, 255, 0.12);
    }
    .shulko-hero-icon {
        font-size: 40px;
        filter: drop-shadow(0 0 12px rgba(108, 99, 255, 0.7));
    }
    .shulko-hero-title {
        font-size: 22px;
        font-weight: 700;
        color: #E7E8FF;
    }
    .shulko-hero-title span { color: #7C9CFF; }
    .shulko-hero-sub {
        font-size: 13.5px;
        color: #B9C2D0;
        margin-top: 6px;
        line-height: 1.55;
    }

    /* -- glowing "live" orb, reused by the empty state and the loader -- */
    @keyframes shulko-pulse {
        0%, 100% { transform: scale(0.85); opacity: 0.7; }
        50%      { transform: scale(1.15); opacity: 1;   }
    }
    @keyframes shulko-float {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        50%      { transform: translateY(-6px) rotate(4deg); }
    }
    .shulko-orb {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        flex-shrink: 0;
        background: radial-gradient(
            circle at 35% 30%, #E0E7FF, #6C63FF 45%, #312E81 100%
        );
        box-shadow: 0 0 22px rgba(108, 99, 255, 0.75);
        animation: shulko-pulse 2.2s ease-in-out infinite;
    }

    /* -- empty-state welcome screen (no conversation yet) -- */
    .shulko-empty {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 56px 12px 28px;
    }
    .shulko-orb-lg {
        width: 84px;
        height: 84px;
        margin-bottom: 22px;
        animation:
            shulko-pulse 3s ease-in-out infinite,
            shulko-float 3.6s ease-in-out infinite;
    }
    .shulko-empty-title {
        font-size: 22px;
        font-weight: 600;
        color: #EDEFF3;
        margin-bottom: 28px;
    }

    /* -- inline "thinking" loader, replaces Streamlit's default spinner -- */
    .shulko-loader {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 4px;
        color: #B9C2D0;
        font-size: 14px;
    }
    .shulko-loader .shulko-orb {
        width: 20px;
        height: 20px;
        animation-duration: 1.1s;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# TRANSLATIONS
# ------------------------------------------------------------------

TEXT = {
    "en": {
        "rate_card": "Statutory rates for this heading",
        "rate_note": (
            "You asked about a code, not a shipment — so these are "
            "the rates, not an amount. Attach an invoice for taka figures."
        ),
        "tti": "Total tax incidence",
        "rate": "Rate",
        "title": "Shulko",
        "tagline": "Bangladesh import landed-cost agent",
        "greeting": (
            "Attach a commercial invoice in the sidebar, or ask me "
            "about duty on an HS code.\n\n"
            "Try: *what is the import duty on polypropylene granules?*"
        ),
        "shipment": "Shipment costs",
        "shipment_help": (
            "Your invoice prices the goods only (FOB). Bangladesh "
            "customs charges duty on the landed value, so freight and "
            "insurance are added before any rate applies. Enter your "
            "own figures — these are just starting values."
        ),
        "freight": "Freight (BDT)",
        "freight_help": (
            "What you pay to ship the goods to Chattogram. Any amount."
        ),
        "insurance": "Insurance (BDT)",
        "insurance_help": (
            "Marine insurance on the shipment. Enter 0 if uninsured."
        ),
        "importer": "Importer type",
        "importer_help": (
            "Commercial importers pay Advance Tax (AT); "
            "industrial importers do not."
        ),
        "analyze": "Analyze invoice",
        "note": "Anything to add? (optional)",
        "note_ph": "e.g. these are for a VAT-registered cable factory",
        "ready": "Ready — press Analyze invoice",
        "commercial": "Commercial",
        "industrial": "Industrial",
        "invoice": "Commercial invoice",
        "clear": "＋ New chat",
        "ask": "Attach an invoice, or ask about a tariff code…",
        "av": "Assessable value",
        "tax": "Total tax",
        "landed": "Landed cost",
        "why": "Why this HS code?",
        "alternatives": "Other candidates the classifier considered",
        "review": "Needs human review before filing",
        "verification": "Classification verification",
        "verification_pass": "Verification passed",
        "verification_fail": (
            "This classification could not be fully verified."
        ),
        "verification_unconfirmed": (
            "The HS code is a classifier result, but it is not "
            "confirmed for filing."
        ),
        "evidence_missing": (
            "The classifier did not provide sufficient legal evidence "
            "for this classification."
        ),
        "regulatory": "Regulatory advisories",
        "download": "Download as CSV",
        "thinking": "Running the pipeline…",
        "offline": (
            "Backend not reachable. Start it with: "
            "uvicorn backend.main:app --reload"
        ),
        "confidence": "confidence",
        "no_items": "No line items could be read from that document.",
        "grounding_via": (
            "Regulatory advisories above were retrieved live via {engine}."
        ),
        "grounding_clear": (
            "Checked live via {engine}: no additional Bangladesh "
            "import restriction or certificate requirement was found "
            "for these codes."
        ),
        "grounding_off": (
            "The live regulatory check could not run (no search key, "
            "or the daily API quota is exhausted), so this run carries "
            "no advisories. Absence here does not mean none apply — "
            "verify SRO and certificate requirements separately."
        ),
        "auth_error_generic": "Something went wrong. Please try again.",
        "login": "Log in",
        "signup": "Sign up",
        "email": "Email",
        "password": "Password",
        "password_help": "At least 8 characters.",
        "display_name": "Display name (optional)",
        "login_button": "Log in",
        "signup_button": "Create account",
        "logout": "Log out",
        "need_account": "Need an account? Sign up",
        "have_account": "Already have an account? Log in",
        "app_intro": (
            "Bangladesh import landed-cost agent — log in to analyze "
            "invoices and keep your history."
        ),
        "history": "Recent documents",
        "no_history": (
            "No analyses yet — your history will appear here."
        ),
        "reopen": "Reopen",
        "new_analysis": "＋ New analysis",
        "needs_review_badge": "⚠ review",
        "listen": "🔊 Listen to summary",
        "ask_hs_pill": "Ask about an HS code",
        "attach_pill": "Attach an invoice",
        "attach_pill_help": "Use the 📎 icon in the chat box below",
        "how_pill": "How this works",
        "how_toast": "Attach an invoice, or type a question below",
        "sample_question": (
            "what is the import duty on polypropylene granules?"
        ),
        "extracted_codes": "Extracted HS codes",
        "menu_hide": "☰ Hide menu",
        "menu_show": "☰ Show menu",
        "feat_hs_title": "HS Code Lookup",
        "feat_hs_sub": "Top 3 suggestions",
        "feat_duty_title": "Duty Calculation",
        "feat_duty_sub": "Full tax breakdown",
        "feat_reg_title": "Regulatory Check",
        "feat_reg_sub": "SROs, bans & certificates",
    },

    "bn": {
        "rate_card": "এই heading-এর সরকারি হার",
        "rate_note": (
            "আপনি একটা code নিয়ে জিজ্ঞেস করেছেন, চালান নিয়ে নয় — "
            "তাই এখানে হার দেখানো হচ্ছে, টাকার অঙ্ক নয়। "
            "টাকার হিসাব পেতে invoice দিন।"
        ),
        "tti": "মোট করভার",
        "rate": "হার",
        "title": "শুল্ক",
        "tagline": "বাংলাদেশের আমদানি খরচ নির্ণয়ক",
        "greeting": (
            "পাশের প্যানেলে commercial invoice দিন, অথবা কোনো HS code-এর "
            "শুল্ক জিজ্ঞেস করুন।\n\n"
            "যেমন: *পলিপ্রোপিলিন দানার আমদানি শুল্ক কত?*"
        ),
        "shipment": "চালানের খরচ",
        "shipment_help": (
            "Invoice-এ শুধু পণ্যের দাম থাকে (FOB)। কিন্তু কাস্টমস শুল্ক "
            "বসায় বন্দরে পৌঁছানো মোট মূল্যের উপর — তাই জাহাজ ভাড়া আর "
            "বীমা আগে যোগ হয়। আপনার নিজের অঙ্ক বসান; এগুলো শুধু শুরুর মান।"
        ),
        "freight": "জাহাজ ভাড়া (টাকা)",
        "freight_help": (
            "চট্টগ্রাম পর্যন্ত মাল আনতে যা খরচ। যেকোনো অঙ্ক দিতে পারেন।"
        ),
        "insurance": "বীমা (টাকা)",
        "insurance_help": (
            "চালানের সামুদ্রিক বীমা। বীমা না থাকলে ০ দিন।"
        ),
        "importer": "আমদানিকারকের ধরন",
        "importer_help": (
            "বাণিজ্যিক আমদানিকারক Advance Tax (AT) দেয়, শিল্প দেয় না।"
        ),
        "analyze": "হিসাব করুন",
        "note": "কিছু যোগ করতে চান? (ঐচ্ছিক)",
        "note_ph": "যেমন: এগুলো VAT-নিবন্ধিত ক্যাবল কারখানার জন্য",
        "ready": "প্রস্তুত — \"হিসাব করুন\" চাপুন",
        "commercial": "বাণিজ্যিক",
        "industrial": "শিল্প",
        "invoice": "কমার্শিয়াল ইনভয়েস",
        "clear": "＋ নতুন চ্যাট",
        "ask": "Invoice দিন, বা কোনো tariff code জিজ্ঞেস করুন…",
        "av": "নিরূপিত মূল্য",
        "tax": "মোট কর",
        "landed": "সর্বমোট খরচ",
        "why": "এই HS code কেন?",
        "alternatives": "বিবেচনায় আসা অন্যান্য code",
        "review": "দাখিলের আগে যাচাই প্রয়োজন",
        "verification": "শ্রেণিবিন্যাস যাচাই",
        "verification_pass": "যাচাই সফল হয়েছে",
        "verification_fail": (
            "এই classification পুরোপুরি যাচাই করা যায়নি।"
        ),
        "verification_unconfirmed": (
            "এটি classifier-এর ফলাফল, কিন্তু দাখিলের জন্য "
            "এখনো নিশ্চিত নয়।"
        ),
        "evidence_missing": (
            "এই classification-এর পক্ষে যথেষ্ট আইনি প্রমাণ "
            "classifier দেয়নি।"
        ),
        "regulatory": "নিয়মকানুন সংক্রান্ত সতর্কতা",
        "download": "CSV ডাউনলোড",
        "thinking": "প্রক্রিয়া চলছে…",
        "offline": (
            "ব্যাকএন্ড চলছে না। চালু করুন: "
            "uvicorn backend.main:app --reload"
        ),
        "confidence": "নিশ্চয়তা",
        "no_items": "এই ডকুমেন্ট থেকে কোনো পণ্য পড়া যায়নি।",
        "grounding_via": (
            "উপরের নিয়মকানুন সংক্রান্ত তথ্য {engine} দিয়ে "
            "সরাসরি ইন্টারনেট থেকে আনা হয়েছে।"
        ),
        "grounding_clear": (
            "{engine} দিয়ে যাচাই করা হয়েছে — এই code-গুলোর জন্য "
            "অতিরিক্ত কোনো আমদানি নিষেধাজ্ঞা বা সার্টিফিকেটের "
            "প্রয়োজন পাওয়া যায়নি।"
        ),
        "grounding_off": (
            "সরাসরি নিয়মকানুন যাচাই করা যায়নি (search key নেই, "
            "অথবা দৈনিক API কোটা শেষ), তাই এই ফলাফলে কোনো সতর্কতা "
            "নেই। এর মানে কোনো নিয়ম নেই তা নয় — SRO আর "
            "সার্টিফিকেটের শর্ত আলাদাভাবে যাচাই করুন।"
        ),
        "login": "লগ ইন",
        "signup": "নতুন অ্যাকাউন্ট",
        "email": "ইমেইল",
        "password": "পাসওয়ার্ড",
        "password_help": "কমপক্ষে ৮ অক্ষর।",
        "display_name": "নাম (ঐচ্ছিক)",
        "login_button": "লগ ইন করুন",
        "signup_button": "অ্যাকাউন্ট তৈরি করুন",
        "logout": "লগ আউট",
        "need_account": "অ্যাকাউন্ট নেই? নতুন অ্যাকাউন্ট তৈরি করুন",
        "have_account": "আগে থেকেই অ্যাকাউন্ট আছে? লগ ইন করুন",
        "app_intro": (
            "বাংলাদেশের আমদানি খরচ নির্ণয়ক — invoice বিশ্লেষণ "
            "করতে ও আপনার history রাখতে লগ ইন করুন।"
        ),
        "history": "সাম্প্রতিক ডকুমেন্ট",
        "no_history": (
            "এখনো কোনো বিশ্লেষণ নেই — এখানে আপনার history দেখা যাবে।"
        ),
        "reopen": "আবার দেখুন",
        "new_analysis": "＋ নতুন বিশ্লেষণ",
        "needs_review_badge": "⚠ review",
        "listen": "🔊 সারাংশ শুনুন",
        "ask_hs_pill": "শুল্ক হার জানুন",
        "attach_pill": "ইনভয়েস যোগ করুন",
        "attach_pill_help": "নিচের চ্যাট বক্সের 📎 আইকনে ক্লিক করুন",
        "how_pill": "কীভাবে কাজ করে",
        "how_toast": "ইনভয়েস attach করুন অথবা নিচে প্রশ্ন লিখুন",
        "sample_question": "পলিপ্রোপিলিন দানার আমদানি শুল্ক কত?",
        "extracted_codes": "শনাক্ত হওয়া HS code",
        "menu_hide": "☰ মেনু লুকান",
        "menu_show": "☰ মেনু দেখান",
        "feat_hs_title": "HS Code সন্ধান",
        "feat_hs_sub": "সেরা ৩টা প্রস্তাবনা",
        "feat_duty_title": "শুল্ক হিসাব",
        "feat_duty_sub": "সম্পূর্ণ কর বিবরণ",
        "feat_reg_title": "নিয়মকানুন যাচাই",
        "feat_reg_sub": "SRO, নিষেধাজ্ঞা ও সার্টিফিকেট",
    },
}


def t(key: str, lang: str) -> str:
    return TEXT.get(
        lang,
        TEXT["en"],
    ).get(
        key,
        TEXT["en"].get(key, key),
    )


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

@st.cache_data(ttl=15)
def health() -> dict | None:
    try:
        return requests.get(
            f"{API}/health",
            timeout=5,
        ).json()
    except Exception:
        return None


def auth_headers() -> dict:
    token = st.session_state.get("token")

    if token:
        return {
            "Authorization": f"Bearer {token}"
        }

    return {}


def _auth_request(
    path: str,
    payload: dict,
) -> tuple[dict | None, str | None]:

    try:
        response = requests.post(
            f"{API}{path}",
            json=payload,
            timeout=15,
        )

    except Exception:
        return None, "offline"

    if response.status_code >= 400:
        try:
            return None, response.json().get(
                "detail",
                "error",
            )
        except Exception:
            return None, "error"

    return response.json(), None


def _error_detail(response: requests.Response) -> str:
    try:
        return response.json().get(
            "detail",
            response.text,
        )
    except Exception:
        return (
            f"{response.status_code} "
            f"{response.reason}"
        )


def taka(value: float) -> str:
    return f"৳{value:,.2f}"


def _clean_reasoning(text: str | None) -> str:
    """
    Remove meaningless model output such as:
        -
        *
        >
        > -
        * *
    """

    if not text:
        return ""

    cleaned = str(text).strip()

    # Remove simple markdown markers.
    stripped = cleaned.replace(
        "*", ""
    ).replace(
        ">", ""
    ).replace(
        "-", ""
    ).strip()

    if not stripped:
        return ""

    return cleaned


@contextlib.contextmanager
def _thinking(lang: str):
    """Streamlit-এর default spinner লুকিয়ে, নিজস্ব glowing 'live' loader।"""
    placeholder = st.empty()
    placeholder.markdown(
        f"""
        <div class="shulko-loader">
          <div class="shulko-orb"></div>
          <span>{t("thinking", lang)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        placeholder.empty()


# ------------------------------------------------------------------
# GROUNDING
# ------------------------------------------------------------------

def _render_grounding(
    report: dict,
    lang: str,
) -> None:

    grounding = report.get("grounding") or {}

    if not grounding.get("codes_checked"):
        return

    if not grounding.get("available"):
        st.caption(
            t("grounding_off", lang)
        )
        return

    engine = (
        "Google Search grounding"
        if grounding.get("mechanism") == "google_search"
        else "Tavily search (fallback)"
    )

    key = (
        "grounding_via"
        if report.get("advisories")
        else "grounding_clear"
    )

    st.caption(
        t(key, lang).format(
            engine=engine
        )
    )


# ------------------------------------------------------------------
# VERIFICATION UI
# ------------------------------------------------------------------

def _render_verification(
    result: dict,
    lang: str,
) -> None:

    verification = result.get(
        "verification"
    ) or {}

    if not verification:
        return

    verdict = verification.get(
        "verdict",
        "fail",
    )

    warnings = verification.get(
        "warnings"
    ) or []

    if (
        verdict == "pass"
        and not verification.get("needs_review")
        and not warnings
    ):
        st.success(
            f"✓ {t('verification_pass', lang)}"
        )
        return

    with st.expander(
        f"⚠ {t('verification', lang)}",
        expanded=True,
    ):
        st.warning(
            t("verification_fail", lang)
        )

        st.caption(
            t("verification_unconfirmed", lang)
        )

        if not verification.get(
            "evidence_supported",
            True,
        ):
            st.markdown(
                f"**{t('evidence_missing', lang)}**"
            )

        if warnings:
            for warning in warnings:
                st.markdown(
                    f"- {warning}"
                )


# ------------------------------------------------------------------
# REPORT
# ------------------------------------------------------------------

def render_report(
    report: dict,
    lang: str,
) -> None:

    if not report:
        return

    if report.get("kind") == "refusal":
        st.info(
            report.get("message", "")
        )
        return

    items = report.get("items") or []

    if not items:
        st.warning(
            t("no_items", lang)
        )
        return

    # --------------------------------------------------------------
    # RATE CARD
    # --------------------------------------------------------------

    if report.get("kind") == "rate_card":
        st.caption(
            t("rate_note", lang)
        )

        for result in items:
            _render_rate_card(
                result,
                lang,
            )

        _render_grounding(
            report,
            lang,
        )

        return

    # --------------------------------------------------------------
    # TOTALS
    # --------------------------------------------------------------

    totals = report.get(
        "totals"
    ) or {}

    c1, c2, c3 = st.columns(3)

    c1.metric(
        t("av", lang),
        taka(totals.get("av", 0)),
    )

    c2.metric(
        t("tax", lang),
        taka(totals.get("tti", 0)),
        f"{report.get('effective_rate', 0)}%",
    )

    c3.metric(
        t("landed", lang),
        taka(totals.get("landed_cost", 0)),
    )

    if report.get("needs_review"):
        st.warning(
            f"⚠ {t('review', lang)}"
        )

    if st.button(t("listen", lang), key=f"listen-{id(report)}"):
        st.audio(
            speak(build_plain_summary(report, lang), lang),
            format="audio/mp3",
            autoplay=True,
        )

    # --------------------------------------------------------------
    # ITEMS
    # --------------------------------------------------------------

    for result in items:
        _render_item(
            result,
            lang,
        )

    # --------------------------------------------------------------
    # REGULATORY ADVISORIES
    # --------------------------------------------------------------

    advisories = (
        report.get("advisories")
        or []
    )

    if advisories:
        st.markdown(
            f"**{t('regulatory', lang)}**"
        )

        for advisory in advisories:
            message = advisory.get(
                "message",
                "",
            )

            kind = advisory.get(
                "kind",
                "info",
            )

            source_url = advisory.get(
                "source_url"
            )

            if source_url:
                st.markdown(
                    f"- {message}  \n"
                    f"  [{kind}]({source_url})"
                )
            else:
                st.markdown(
                    f"- {message}  \n"
                    f"  [{kind}]"
                )

    _render_grounding(
        report,
        lang,
    )

    # --------------------------------------------------------------
    # CSV
    # --------------------------------------------------------------

    st.download_button(
        t("download", lang),
        data=_to_csv(items),
        file_name="shulko-landed-cost.csv",
        mime="text/csv",
        key=f"dl-{id(report)}",
    )


# ------------------------------------------------------------------
# RATE CARD
# ------------------------------------------------------------------

def _render_rate_card(
    result: dict,
    lang: str,
) -> None:

    item = result.get(
        "item",
        {},
    )

    candidates = (
        result.get("candidates")
        or []
    )

    rates = (
        result.get("rates_pct")
        or {}
    )

    top = (
        candidates[0]
        if candidates
        else None
    )

    header = item.get(
        "description",
        "item",
    )

    if result.get(
        "chosen_hs_code"
    ):
        header = (
            f"{result['chosen_hs_code']} · "
            f"{header}"
        )

    with st.container(
        border=True
    ):

        st.markdown(
            f"**{header}**"
        )

        if top:
            confidence = min(
                max(
                    float(
                        top.get(
                            "confidence",
                            0.0,
                        )
                    ),
                    0.0,
                ),
                1.0,
            )

            pct = int(
                confidence * 100
            )

            st.caption(
                f"{top.get('heading_text', '')} "
                f"— {pct}% "
                f"{t('confidence', lang)}"
            )

        if rates:
            st.metric(
                t("tti", lang),
                f"{rates.get('total_tax_incidence', 0)}%",
            )

            st.dataframe(
                pd.DataFrame({
                    "Component": [
                        "CD",
                        "RD",
                        "SD",
                        "VAT",
                        "AIT",
                        "AT",
                    ],
                    t("rate", lang): [
                        f"{rates.get(k, 0)}%"
                        for k in (
                            "cd",
                            "rd",
                            "sd",
                            "vat",
                            "ait",
                            "at",
                        )
                    ],
                }),
                hide_index=True,
                width="stretch",
            )

        # Ordinary warnings only.
        for warning in (
            result.get("warnings")
            or []
        ):
            st.warning(warning)

        _render_verification(
            result,
            lang,
        )

        if top:
            with st.expander(
                t("why", lang)
            ):
                reasoning = _clean_reasoning(
                    top.get("reasoning")
                )

                if reasoning:
                    st.markdown(
                        reasoning
                    )
                else:
                    st.caption(
                        "No detailed reasoning was returned."
                    )

                for evidence in (
                    top.get("evidence")
                    or []
                ):
                    source = evidence.get(
                        "source",
                        "Source",
                    )

                    quote = evidence.get(
                        "quote",
                        "",
                    )

                    if not quote:
                        continue

                    mark = (
                        "✓"
                        if evidence.get(
                            "supports",
                            True,
                        )
                        else "✗"
                    )

                    st.markdown(
                        f"> {mark} "
                        f"**{source}** — "
                        f"“{quote}”"
                    )

                if len(candidates) > 1:
                    st.markdown(
                        f"**{t('alternatives', lang)}**"
                    )

                    for candidate in candidates[1:]:
                        confidence = min(
                            max(
                                float(
                                    candidate.get(
                                        "confidence",
                                        0.0,
                                    )
                                ),
                                0.0,
                            ),
                            1.0,
                        )

                        st.markdown(
                            f"- `{candidate['hs_code']}` "
                            f"({int(confidence * 100)}%) "
                            f"{candidate.get('heading_text', '')}"
                        )


# ------------------------------------------------------------------
# NORMAL ITEM
# ------------------------------------------------------------------

def _render_item(
    result: dict,
    lang: str,
) -> None:

    item = result.get(
        "item",
        {},
    )

    candidates = (
        result.get("candidates")
        or []
    )

    duty = (
        result.get("duty")
        or {}
    )

    top = (
        candidates[0]
        if candidates
        else None
    )

    header = item.get(
        "description",
        "item",
    )

    if result.get(
        "chosen_hs_code"
    ):
        header = (
            f"{result['chosen_hs_code']} · "
            f"{header}"
        )

    with st.container(
        border=True
    ):

        st.markdown(
            f"**{header}**"
        )

        # ----------------------------------------------------------
        # CLASSIFIER RESULT
        # ----------------------------------------------------------

        if top:

            confidence = min(
                max(
                    float(
                        top.get(
                            "confidence",
                            0.0,
                        )
                    ),
                    0.0,
                ),
                1.0,
            )

            pct = int(
                confidence * 100
            )

            st.caption(
                f"{top.get('heading_text', '')} "
                f"— {pct}% "
                f"{t('confidence', lang)}"
            )

            # No st.progress().
            st.caption(
                f"{pct}%"
            )

        # ----------------------------------------------------------
        # DUTY -- plain-language labels, not CD/RD/SD/AIT/AT jargon
        # ----------------------------------------------------------

        if duty:
            plain_labels = {
                "en": ["Import Duty", "Extra Duty", "Sales Tax",
                       "VAT", "Advance Income Tax", "Advance Tax", "Total"],
                "bn": ["আমদানি শুল্ক", "অতিরিক্ত শুল্ক", "সম্পূরক কর",
                       "ভ্যাট", "অগ্রিম আয়কর", "অগ্রিম কর", "সর্বমোট"],
            }
            labels = plain_labels.get(lang, plain_labels["en"])

            st.dataframe(
                pd.DataFrame({
                    ("Item" if lang == "en" else "খাত"): labels,
                    ("Amount (BDT)" if lang == "en" else "টাকা"): [
                        duty.get(k, 0)
                        for k in ("cd", "rd", "sd", "vat", "ait", "at", "tti")
                    ],
                }),
                hide_index=True,
                width="stretch",
            )

        # ----------------------------------------------------------
        # CLASSIFICATION WARNINGS
        # ----------------------------------------------------------

        for warning in (
            result.get("warnings")
            or []
        ):
            st.warning(warning)

        # ----------------------------------------------------------
        # VERIFICATION
        # ----------------------------------------------------------

        _render_verification(
            result,
            lang,
        )

        # ----------------------------------------------------------
        # WHY THIS CODE?
        # ----------------------------------------------------------

        if top:
            with st.expander(
                t("why", lang)
            ):

                reasoning = _clean_reasoning(
                    top.get("reasoning")
                )

                if reasoning:
                    st.markdown(
                        reasoning
                    )
                else:
                    st.caption(
                        "No detailed reasoning was returned."
                    )

                evidence_list = (
                    top.get("evidence")
                    or []
                )

                for evidence in evidence_list:

                    source = evidence.get(
                        "source",
                        "Source",
                    )

                    quote = evidence.get(
                        "quote",
                        "",
                    )

                    if not quote:
                        continue

                    mark = (
                        "✓"
                        if evidence.get(
                            "supports",
                            True,
                        )
                        else "✗"
                    )

                    st.markdown(
                        f"> {mark} "
                        f"**{source}** — "
                        f"“{quote}”"
                    )

                # --------------------------------------------------
                # ALTERNATIVES
                # --------------------------------------------------

                if len(candidates) > 1:

                    st.markdown(
                        f"**{t('alternatives', lang)}**"
                    )

                    for candidate in candidates[1:]:

                        confidence = min(
                            max(
                                float(
                                    candidate.get(
                                        "confidence",
                                        0.0,
                                    )
                                ),
                                0.0,
                            ),
                            1.0,
                        )

                        st.markdown(
                            f"- `{candidate['hs_code']}` "
                            f"({int(confidence * 100)}%) "
                            f"{candidate.get('heading_text', '')}"
                        )


# ------------------------------------------------------------------
# CSV
# ------------------------------------------------------------------

def _to_csv(
    items: list[dict],
) -> bytes:

    rows = []

    for result in items:

        duty = (
            result.get("duty")
            or {}
        )

        candidates = (
            result.get("candidates")
            or []
        )

        top = (
            candidates[0]
            if candidates
            else {}
        )

        verification = (
            result.get("verification")
            or {}
        )

        rows.append({
            "description": (
                result.get(
                    "item",
                    {},
                ).get(
                    "description",
                    "",
                )
            ),

            "hs_code": result.get(
                "chosen_hs_code",
                "",
            ),

            "confidence": top.get(
                "confidence",
                0,
            ),

            **{
                k: duty.get(k, 0)
                for k in (
                    "av",
                    "cd",
                    "rd",
                    "sd",
                    "vat",
                    "ait",
                    "at",
                    "tti",
                    "landed_cost",
                )
            },

            "verification_verdict": (
                verification.get(
                    "verdict",
                    "",
                )
            ),

            "verification_needs_review": (
                verification.get(
                    "needs_review",
                    False,
                )
            ),

            "verification_warnings": (
                " | ".join(
                    verification.get(
                        "warnings",
                        [],
                    )
                )
            ),

            "needs_review": result.get(
                "needs_review",
                False,
            ),
        })

    buffer = io.StringIO()

    pd.DataFrame(
        rows
    ).to_csv(
        buffer,
        index=False,
    )

    return buffer.getvalue().encode(
        "utf-8"
    )


# ------------------------------------------------------------------
# CHAT SHORTCUTS -- summarize / HS-code-only / attach+ask, all handled
# without spending a fresh backend call where the answer is already in
# the session.
# ------------------------------------------------------------------

SUMMARY_TRIGGERS = re.compile(
    r"\b(summari[sz]e|summary|tl;?dr)\b|সারাংশ|সংক্ষেপে|সংক্ষিপ্ত",
    re.I,
)

HSCODE_ONLY_TRIGGERS = re.compile(
    r"\bhs\s*code|classif(y|ication)\b|\bcode\b",
    re.I,
)

DUTY_TRIGGERS = re.compile(
    r"duty|tax|cost|vat|landed|price|টাকা|শুল্ক|কর|খরচ",
    re.I,
)


def _last_report() -> dict | None:
    """এই কথোপকথনে সবশেষ যে report দেখানো হয়েছে, সেটা ফেরত দেয়।"""
    for message in reversed(st.session_state.messages):
        if message.get("report") and message["report"].get("totals"):
            return message["report"]
    return None


def _plain_summary(report: dict, lang: str) -> str:
    """Business owner বুঝবে এমন ২ লাইনের সারাংশ, backend না ডেকেই।"""
    totals = report.get("totals") or {}
    tax = totals.get("tti", 0)
    landed = totals.get("landed_cost", 0)
    needs_review = report.get("needs_review")

    names = [
        (r.get("item") or {}).get("description", "")
        for r in (report.get("items") or [])
    ]
    names_text = ", ".join(n for n in names if n) or (
        "আপনার পণ্য" if lang == "bn" else "your goods"
    )

    if lang == "bn":
        text = (
            f"{names_text}-এর জন্য মোট কর {tax:,.0f} টাকা, "
            f"সব খরচসহ মোট {landed:,.0f} টাকা।"
        )
        text += (
            " এটি জমা দেওয়ার আগে একজন মানুষকে দিয়ে যাচাই করিয়ে নিন।"
            if needs_review else " এটি যাচাই করা হয়েছে।"
        )
    else:
        text = (
            f"For {names_text}, total tax is {tax:,.0f} taka, "
            f"landed cost is {landed:,.0f} taka."
        )
        text += (
            " Please have a person check this before filing."
            if needs_review else " This has been verified."
        )
    return text


def _hs_codes_only_view(report: dict, lang: str) -> None:
    """ইউজার শুধু HS code চাইলে, পুরো duty report না দেখিয়ে ছোট তালিকা।"""
    items = report.get("items") or []
    if not items:
        st.warning(t("no_items", lang))
        return

    st.markdown(f"**{t('extracted_codes', lang)}:**")

    for r in items:
        item = r.get("item") or {}
        code = r.get("chosen_hs_code") or "—"
        st.markdown(f"- **`{code}`** — {item.get('description', '')}")


def _run_duty_question(question: str, lang: str) -> None:
    """Pill বাটন বা chat বক্স -- দুই জায়গা থেকেই একই duty-question flow।"""
    st.session_state.messages.append({"role": "user", "text": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with _thinking(lang):
            try:
                response = requests.post(
                    f"{API}/ask",
                    json={
                        "question": question,
                        "language": lang,
                        "freight": freight,
                        "insurance": insurance,
                        "importer_type": importer_type,
                    },
                    headers=auth_headers(),
                    timeout=TIMEOUT,
                )
                response.raise_for_status()
                payload = response.json()
            except requests.HTTPError:
                st.error(_error_detail(response))
                payload = None
            except Exception as exc:
                st.error(f"{exc}")
                payload = None

        if payload:
            report = payload.get("report")
            render_report(report, lang)
            st.session_state.messages.append({"role": "assistant", "report": report})

    st.rerun()


# ------------------------------------------------------------------
# HERO CARDS
# ------------------------------------------------------------------

def _hero(lang: str) -> None:
    """Login পেজের card -- এটা কীসের agent, বাংলা+ইংরেজি দুটোতেই।"""
    st.markdown(
        """
        <div class="shulko-hero">
          <div class="shulko-hero-icon">🛃</div>
          <div>
            <div class="shulko-hero-title">Shulko <span>শুল্ক</span></div>
            <div class="shulko-hero-sub">
              AI agent for Bangladesh import duty — attach an invoice and
              get the HS code, duty, and landed cost, verified and
              explained in plain language.<br>
              বাংলাদেশের আমদানি শুল্কের AI এজেন্ট — ইনভয়েস দিলে HS code,
              শুল্ক আর সর্বমোট খরচ পাবেন, যাচাই করা ও সহজ ভাষায় ব্যাখ্যাসহ।
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _empty_state(lang: str) -> None:
    """কথোপকথন শুরুর আগে -- বড়, কেন্দ্রে glowing orb + এক লাইন + shortcut।

    Login-এর boxed card থেকে ইচ্ছাকৃতভাবে আলাদা -- এটা কোনো card না,
    খোলা জায়গায় বসে, ঠিক প্রথমবার একটা AI agent খুললে যেমন দেখায়।
    """
    heading = (
        "ইনভয়েস দিন, শুল্কের হিসাব নিন"
        if lang == "bn" else
        "Attach an invoice, get your duty answer"
    )
    st.markdown(
        f"""
        <div class="shulko-empty">
          <div class="shulko-orb shulko-orb-lg"></div>
          <div class="shulko-empty-title">{heading}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    p1, p2, p3 = st.columns(3)

    with p1:
        if st.button(
            f"🧾 {t('ask_hs_pill', lang)}",
            width="stretch",
            key="pill_ask",
        ):
            _run_duty_question(t("sample_question", lang), lang)

    with p2:
        st.button(
            f"📎 {t('attach_pill', lang)}",
            width="stretch",
            disabled=True,
            help=t("attach_pill_help", lang),
            key="pill_attach",
        )

    with p3:
        if st.button(
            f"❓ {t('how_pill', lang)}",
            width="stretch",
            key="pill_how",
        ):
            st.toast(t("how_toast", lang))

    f1, f2, f3 = st.columns(3)

    features = [
        (f1, "📄", "feat_hs_title", "feat_hs_sub", "0s"),
        (f2, "🧮", "feat_duty_title", "feat_duty_sub", "0.3s"),
        (f3, "🛡️", "feat_reg_title", "feat_reg_sub", "0.6s"),
    ]

    for col, icon, title_key, sub_key, delay in features:
        with col:
            st.markdown(
                f"""
                <div class="shulko-feature">
                  <div class="shulko-feature-icon" style="animation-delay:{delay}">
                    {icon}
                  </div>
                  <div>
                    <div class="shulko-feature-title">{t(title_key, lang)}</div>
                    <div class="shulko-feature-sub">{t(sub_key, lang)}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ------------------------------------------------------------------
# AUTH GATE
# ------------------------------------------------------------------

st.session_state.setdefault(
    "token",
    None,
)

st.session_state.setdefault(
    "user",
    None,
)

st.session_state.setdefault(
    "auth_lang",
    "en",
)


if not st.session_state.token:

    _, lang_col = st.columns(
        [3, 1]
    )

    with lang_col:
        picked = st.segmented_control(
            "lang",
            ["en", "bn"],
            default=st.session_state.auth_lang,
            format_func=lambda x: "EN" if x == "en" else "বাং",
            label_visibility="collapsed",
            key="gate_lang_switch",
        )

    st.session_state.auth_lang = picked or st.session_state.auth_lang

    gate_lang = (
        st.session_state.auth_lang
    )

    _hero(gate_lang)

    st.caption(
        t("app_intro", gate_lang)
    )

    tab_login, tab_signup = st.tabs([
        t("login", gate_lang),
        t("signup", gate_lang),
    ])

    with tab_login:

        with st.form(
            "login_form"
        ):

            email = st.text_input(
                t("email", gate_lang)
            )

            password = st.text_input(
                t("password", gate_lang),
                type="password",
            )

            go = st.form_submit_button(
                t("login_button", gate_lang),
                type="primary",
            )

        if go:

            data, err = _auth_request(
                "/auth/login",
                {
                    "email": email,
                    "password": password,
                },
            )

            if err == "offline":
                st.error(
                    t("offline", gate_lang)
                )

            elif err:
                st.error(err)

            else:
                st.session_state.token = (
                    data["access_token"]
                )

                st.session_state.user = (
                    data["user"]
                )

                st.rerun()

    with tab_signup:

        with st.form(
            "signup_form"
        ):

            s_email = st.text_input(
                t("email", gate_lang),
                key="signup_email",
            )

            s_password = st.text_input(
                t("password", gate_lang),
                type="password",
                help=t(
                    "password_help",
                    gate_lang,
                ),
                key="signup_password",
            )

            s_name = st.text_input(
                t("display_name", gate_lang),
                key="signup_name",
            )

            go = st.form_submit_button(
                t(
                    "signup_button",
                    gate_lang,
                ),
                type="primary",
            )

        if go:

            data, err = _auth_request(
                "/auth/signup",
                {
                    "email": s_email,
                    "password": s_password,
                    "display_name": s_name,
                },
            )

            if err == "offline":
                st.error(
                    t("offline", gate_lang)
                )

            elif err:
                st.error(err)

            else:
                st.session_state.token = (
                    data["access_token"]
                )

                st.session_state.user = (
                    data["user"]
                )

                st.rerun()

    st.stop()


# ------------------------------------------------------------------
# LANGUAGE SWITCH -- top-right of the main area, not the sidebar.
# Computed before the sidebar block so the sidebar's own labels use it.
# ------------------------------------------------------------------

st.session_state.setdefault("sidebar_open", True)

top_menu, top_l, top_status, top_lang = st.columns([1.3, 2.7, 1.3, 1])

with top_menu:
    _menu_label = (
        t("menu_hide", st.session_state.auth_lang)
        if st.session_state.sidebar_open
        else t("menu_show", st.session_state.auth_lang)
    )
    if st.button(_menu_label, key="sidebar_toggle"):
        st.session_state.sidebar_open = not st.session_state.sidebar_open
        st.rerun()

with top_status:
    _status = health()
    _online = _status is not None
    st.markdown(
        f"""
        <div class="shulko-status-pill">
          <span class="shulko-status-dot{'' if _online else ' off'}"></span>
          {"Online" if _online else "Offline"}
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_lang:
    picked_lang = st.segmented_control(
        "lang",
        ["en", "bn"],
        default=st.session_state.auth_lang,
        format_func=lambda x: "EN" if x == "en" else "বাং",
        label_visibility="collapsed",
        key="main_lang_switch",
    )

lang = picked_lang or st.session_state.auth_lang
st.session_state.auth_lang = lang

if not st.session_state.sidebar_open:
    st.markdown(
        '<style>[data-testid="stSidebar"] { display: none !important; }</style>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------

with st.sidebar:

    st.markdown(
        "### 🛃 Shulko / শুল্ক"
    )

    status = health()

    if status is None:

        st.error(
            t("offline", lang)
        )

    else:

        indexed = status.get(
            "indexed",
            {},
        )

        st.caption(
            f"{status.get('retrieval_mode', '?')} · "
            f"{indexed.get('tariff_lines', 0):,} tariff lines · "
            f"{indexed.get('chapter_notes', 0)} notes"
        )

        if not status.get(
            "langsmith_tracing"
        ):
            st.caption(
                "⚠ LangSmith tracing off"
            )

    st.divider()

    st.markdown(
        f"**{t('shipment', lang)}**"
    )

    st.caption(
        t("shipment_help", lang)
    )

    freight = st.number_input(
        t("freight", lang),
        min_value=0.0,
        value=5000.0,
        step=500.0,
        help=t(
            "freight_help",
            lang,
        ),
    )

    insurance = st.number_input(
        t("insurance", lang),
        min_value=0.0,
        value=1000.0,
        step=100.0,
        help=t(
            "insurance_help",
            lang,
        ),
    )

    importer_type = st.selectbox(
        t("importer", lang),
        [
            "commercial",
            "industrial",
        ],
        format_func=lambda x: t(
            x,
            lang,
        ),
        help=t(
            "importer_help",
            lang,
        ),
    )

    st.divider()

    if st.button(
        t("clear", lang),
        width="stretch",
    ):

        st.session_state.messages = []
        st.session_state.handled = None
        st.session_state.viewing_run_id = None

        st.rerun()

    # --------------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------------

    st.divider()

    st.markdown(
        f"**{t('history', lang)}**"
    )

    try:

        response = requests.get(
            f"{API}/runs",
            headers=auth_headers(),
            timeout=10,
        )

        response.raise_for_status()

        past_runs = response.json()

    except Exception:
        past_runs = []

    if not past_runs:

        st.caption(
            t("no_history", lang)
        )

    else:

        for run in past_runs[:15]:

            when = (
                run.get(
                    "created_at",
                    "",
                )[:10]
            )

            badge = (
                f" {t('needs_review_badge', lang)}"
                if run.get("needs_review")
                else ""
            )

            if st.button(
                f"📄 {run['filename']}  \n"
                f"{when}{badge}",
                key=f"hist-{run['id']}",
                width="stretch",
            ):

                st.session_state.viewing_run_id = (
                    run["id"]
                )

                st.rerun()

    # --------------------------------------------------------------
    # ACCOUNT -- deliberately last, so it renders at the bottom of the
    # sidebar's content instead of the top. Streamlit has no reliable,
    # version-safe way to pin it to the screen's actual bottom edge
    # (that needs fragile CSS keyed to internal testids) -- this gets
    # "below everything else" without the risk of it breaking on the
    # next Streamlit upgrade.
    # --------------------------------------------------------------

    st.divider()

    prof_col, name_col = st.columns([1, 4])

    with prof_col:
        st.markdown(
            "<div style='font-size:26px;'>🧑‍💼</div>",
            unsafe_allow_html=True,
        )

    with name_col:
        st.markdown(f"**{st.session_state.user['display_name']}**")

        if st.button(t("logout", lang), key="logout_btn", width="stretch"):
            st.session_state.token = None
            st.session_state.user = None
            st.session_state.messages = []
            st.session_state.viewing_run_id = None
            st.rerun()


# ------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------

st.session_state.setdefault(
    "messages",
    [],
)

st.session_state.setdefault(
    "handled",
    None,
)

st.session_state.setdefault(
    "viewing_run_id",
    None,
)


# ------------------------------------------------------------------
# REOPENED ANALYSIS
# ------------------------------------------------------------------

if st.session_state.viewing_run_id:

    st.subheader(
        t("title", lang)
    )

    if st.button(
        f"← {t('new_analysis', lang)}"
    ):

        st.session_state.viewing_run_id = None

        st.rerun()

    try:

        response = requests.get(
            f"{API}/runs/"
            f"{st.session_state.viewing_run_id}",
            headers=auth_headers(),
            timeout=10,
        )

        response.raise_for_status()

        past = response.json()

    except Exception as exc:

        st.error(
            f"{exc}"
        )

        past = None

    if past:

        st.caption(
            f"📄 {past['filename']} · "
            f"{(past.get('created_at') or '')[:10]}"
        )

        with st.chat_message(
            "assistant"
        ):
            render_report(
                past.get("report"),
                lang,
            )

    st.stop()


# ------------------------------------------------------------------
# CONVERSATION
# ------------------------------------------------------------------

if not st.session_state.messages:
    _empty_state(lang)


for message in (
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        if message.get("text"):
            st.markdown(
                message["text"]
            )

        if message.get("report"):
            if message.get("view") == "hs_only":
                _hs_codes_only_view(message["report"], lang)
            else:
                render_report(
                    message["report"],
                    lang,
                )


# ------------------------------------------------------------------
# DUTY QUESTION / INVOICE ATTACHMENT
# ------------------------------------------------------------------

chat_value = st.chat_input(
    t("ask", lang),
    accept_file=True,
    file_type=["pdf", "png", "jpg", "jpeg", "webp"],
)

if chat_value:

    prompt = chat_value.text or ""
    attached = chat_value.files[0] if chat_value.files else None

    label = prompt
    if attached:
        label = f"📎 **{attached.name}**" + (f"\n\n{prompt}" if prompt else "")

    st.session_state.messages.append({"role": "user", "text": label})

    with st.chat_message("user"):
        st.markdown(label)

    # --------------------------------------------------- file + instruction
    if attached:

        with st.chat_message("assistant"):

            with _thinking(lang):

                try:
                    response = requests.post(
                        f"{API}/analyze",
                        files={"file": (attached.name, attached.getvalue())},
                        data={
                            "freight": freight,
                            "insurance": insurance,
                            "importer_type": importer_type,
                            "language": lang,
                        },
                        headers=auth_headers(),
                        timeout=TIMEOUT,
                    )
                    response.raise_for_status()
                    payload = response.json()

                except requests.HTTPError:
                    st.error(_error_detail(response))
                    payload = None

                except Exception as exc:
                    st.error(f"{exc}")
                    payload = None

            if payload:

                report = payload.get("report")
                hs_only = bool(
                    HSCODE_ONLY_TRIGGERS.search(prompt)
                    and not DUTY_TRIGGERS.search(prompt)
                )

                if hs_only:
                    _hs_codes_only_view(report, lang)
                else:
                    render_report(report, lang)

                st.session_state.messages.append({
                    "role": "assistant",
                    "report": report,
                    "view": "hs_only" if hs_only else "full",
                })

        st.stop()

    # ------------------------------------------------------------ summary
    last_report = _last_report()

    if SUMMARY_TRIGGERS.search(prompt) and last_report:

        summary_text = _plain_summary(last_report, lang)

        st.session_state.messages.append({"role": "assistant", "text": summary_text})

        with st.chat_message("assistant"):
            st.markdown(summary_text)

        st.stop()

    # ------------------------------------------------------- duty question
    _run_duty_question(prompt, lang)
