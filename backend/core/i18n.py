"""Bilingual UI strings. The target user reads Bangla; the evaluator reads English."""

STRINGS = {
    "en": {
        "app_title": "Shulko — Import Landed-Cost Agent",
        "upload_header": "1. Upload your commercial invoice",
        "review_header": "2. Review extracted items",
        "report_header": "3. Landed-cost report",
        "freight": "Freight (BDT)",
        "insurance": "Insurance (BDT)",
        "importer_type": "Importer type",
        "commercial": "Commercial",
        "industrial": "Industrial",
        "analyze": "Analyze invoice",
        "assessable_value": "Assessable Value",
        "total_tax": "Total Tax Incidence",
        "landed_cost": "Landed Cost",
        "why_this_code": "Why this HS code?",
        "needs_review": "Needs human review",
        "advisories": "Regulatory advisories",
        "download_csv": "Download CSV",
    },
    "bn": {
        "app_title": "শুল্ক — আমদানি খরচ নির্ণয়ক",
        "upload_header": "১. আপনার commercial invoice আপলোড করুন",
        "review_header": "২. বের করা পণ্যগুলো যাচাই করুন",
        "report_header": "৩. মোট খরচের হিসাব",
        "freight": "জাহাজ ভাড়া (টাকা)",
        "insurance": "বীমা (টাকা)",
        "importer_type": "আমদানিকারকের ধরন",
        "commercial": "বাণিজ্যিক",
        "industrial": "শিল্প",
        "analyze": "হিসাব করুন",
        "assessable_value": "নিরূপিত মূল্য",
        "total_tax": "মোট করভার",
        "landed_cost": "সর্বমোট খরচ",
        "why_this_code": "এই HS code কেন?",
        "needs_review": "যাচাই প্রয়োজন",
        "advisories": "নিয়মকানুন সংক্রান্ত সতর্কতা",
        "download_csv": "CSV ডাউনলোড",
    },
}


def t(key: str, lang: str = "en") -> str:
    """Translate a key. Falls back to English, then to the key itself."""
    return STRINGS.get(lang, {}).get(key) or STRINGS["en"].get(key, key)
