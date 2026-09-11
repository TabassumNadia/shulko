"""
Report-কে একটা সহজ, ২-৩ লাইনের বাংলা/ইংরেজি বাক্যে বদলায় এবং শোনায়।

একজন ছোট ব্যবসায়ী "AIT" বা "SD" বোঝেন না। এই ফাইল টেকনিক্যাল ব্রেকডাউন
দেখায় না -- শুধু যে সংখ্যাটা আসলে দরকার (মোট কর) সেটা বলে, আর বলে
এটা যাচাই করা দরকার কিনা।
"""

from __future__ import annotations

import io

from gtts import gTTS


def build_plain_summary(report: dict, lang: str) -> str:
    """Business owner বুঝবে এমন একটা ছোট প্যারাগ্রাফ।"""

    items = report.get("items") or []
    totals = report.get("totals") or {}
    tax = totals.get("tti", 0)
    landed = totals.get("landed_cost", 0)
    needs_review = report.get("needs_review")

    names = [(r.get("item") or {}).get("description", "") for r in items]
    names_text = ", ".join(n for n in names if n) or (
        "আপনার পণ্য" if lang == "bn" else "your goods"
    )

    if lang == "bn":
        text = (
            f"{names_text}-এর জন্য মোট কর আসছে {tax:,.0f} টাকা। "
            f"সব খরচসহ মোট দাঁড়াচ্ছে {landed:,.0f} টাকা।"
        )
        text += (
            " এই হিসাবটি জমা দেওয়ার আগে একজন মানুষকে দিয়ে যাচাই করিয়ে নিন।"
            if needs_review else
            " এই হিসাব যাচাই করা হয়েছে।"
        )
    else:
        text = (
            f"For {names_text}, the total tax comes to {tax:,.0f} taka. "
            f"The full landed cost is {landed:,.0f} taka."
        )
        text += (
            " Please have a person check this before filing."
            if needs_review else
            " This has been verified."
        )

    return text


def speak(text: str, lang: str) -> bytes:
    """লেখা থেকে mp3 bytes -- সরাসরি st.audio()-এ দেওয়া যাবে।"""

    audio = io.BytesIO()
    gTTS(text=text, lang="bn" if lang == "bn" else "en").write_to_fp(audio)
    return audio.getvalue()