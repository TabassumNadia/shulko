"""
Language handling for AI-generated text, in one place.

Two different kinds of text need two different treatments, and mixing
them up is how you get either a broken citation or an English sentence
in an otherwise-Bangla answer:

  1. LLM-generated prose (classification reasoning, verifier warnings,
     regulatory advisories) -- steered with `language_directive()`,
     appended to the relevant agent's system prompt.
  2. Python-generated strings (a threshold check, a "not found" error,
     a fallback message when a call fails) -- these never go through the
     model at all, so steering a prompt cannot touch them. `tr()` looks
     them up from MESSAGES instead.

Deliberately NOT covered by either: HS codes, section/chapter numbers,
statutory rates, dates, filenames, and quoted note text. Those are
translated nowhere in this file, on purpose -- the Verifier checks a
quote against the source note it was pulled from, and a paraphrase
would silently break that check while looking like a translation
improvement.

(This module used to hold a small, unused set of frontend UI strings.
Those now live inline in frontend/app.py's TEXT dict, which is what the
UI actually reads -- nothing imported this file. Repurposed rather than
left to rot as a second, out-of-sync copy of the same idea.)
"""

from __future__ import annotations

LANGUAGE_NAMES = {"en": "English", "bn": "Bengali (Bangla)"}


def language_directive(lang: str) -> str:
    """Appended to an agent's system prompt. Empty for English (the
    prompts are already written in English; a no-op directive would
    just spend tokens for nothing)."""
    if lang not in LANGUAGE_NAMES or lang == "en":
        return ""
    name = LANGUAGE_NAMES[lang]
    return (
        f"\n\nLANGUAGE: Write every explanation, reasoning, and warning in "
        f"natural, fluent {name} -- not a literal machine translation. "
        f"Keep HS codes, section/chapter numbers, statutory rates, monetary "
        f"amounts, dates, and any text you quote verbatim from a source "
        f"exactly as they appear in the source (do not translate a quoted "
        f"note, a citation, or a code). Where a technical term is clearer "
        f"left in English (e.g. an English product or component name), you "
        f"may keep that term in English inside an otherwise-{name} sentence."
    )


MESSAGES: dict[str, dict[str, str]] = {
    "no_line_items": {
        "en": "No line items could be read from the document.",
        "bn": "এই ডকুমেন্ট থেকে কোনো পণ্যের তথ্য (line item) পড়া যায়নি।",
    },
    "no_heading_matched": {
        "en": "No tariff heading matched this item.",
        "bn": "এই পণ্যের জন্য কোনো tariff heading মেলেনি।",
    },
    "classification_failed": {
        "en": "Classification failed for this item: {error}",
        "bn": "এই পণ্যের classification করা যায়নি: {error}",
    },
    "hs_code_not_found": {
        "en": "{code} is not in the loaded tariff schedule.",
        "bn": "{code} বর্তমানে লোড করা tariff schedule-এ নেই।",
    },
    "verification_failed": {
        "en": "Verification could not be completed ({error}). Treat this "
              "classification as unconfirmed.",
        "bn": "যাচাই (verification) সম্পন্ন করা যায়নি ({error})। এই "
              "classification-টিকে এখনও নিশ্চিত নয় বলে ধরে নিন।",
    },
    "no_heading_could_be_matched": {
        "en": "No tariff heading could be matched to this item. It may "
              "fall outside the loaded schedule.",
        "bn": "এই পণ্যের সাথে কোনো tariff heading মেলানো যায়নি। এটি হয়তো "
              "লোড করা schedule-এর বাইরে পড়ে।",
    },
    "confidence_below_threshold": {
        "en": "Confidence {confidence} is below the {threshold} threshold. "
              "A broker should confirm {hs_code} before this is filed.",
        "bn": "নিশ্চয়তা (confidence) {confidence}, যা {threshold} সীমার নিচে। "
              "দাখিল করার আগে একজন broker-কে দিয়ে {hs_code} নিশ্চিত করানো উচিত।",
    },
    "close_runner_up": {
        "en": "{other_code} scores almost as well as {hs_code}. Both are "
              "defensible; the choice changes the duty.",
        "bn": "{hs_code}-এর প্রায় সমান স্কোর পাচ্ছে {other_code}-ও। দুটোই "
              "যুক্তিসঙ্গত — কোনটি বেছে নেওয়া হচ্ছে তার উপর শুল্কের পরিমাণ "
              "নির্ভর করবে।",
    },
    "retrieval_only_fallback": {
        "en": "Retrieval only; the ranking stage returned no usable candidate.",
        "bn": "শুধু retrieval-এর ভিত্তিতে দেওয়া — ranking ধাপ থেকে ব্যবহারযোগ্য "
              "কোনো candidate পাওয়া যায়নি।",
    },
    "notes_available_not_cited": {
        "en": "{notes_available} Section/Chapter Note(s) for Chapter "
              "{chapter} were available to the classifier but none was "
              "cited, so {hs_code} rests on description matching alone.",
        "bn": "Chapter {chapter}-এর জন্য {notes_available}টি Section/Chapter "
              "Note classifier-এর সামনে ছিল, কিন্তু কোনোটিই উদ্ধৃত করা হয়নি — "
              "তাই {hs_code} শুধু বর্ণনার মিলের উপর ভিত্তি করে দেওয়া।",
    },
    "no_note_loaded_for_chapter": {
        "en": "No Section or Chapter Note for Chapter {chapter} is loaded "
              "in the knowledge base, so there was none to cite; {hs_code} "
              "rests on the heading description.{coverage}",
        "bn": "Chapter {chapter}-এর জন্য কোনো Section বা Chapter Note "
              "knowledge base-এ লোড করা নেই, তাই উদ্ধৃত করার মতো কিছু ছিল না; "
              "{hs_code} শুধু heading-এর বর্ণনার উপর ভিত্তি করে দেওয়া।{coverage}",
    },
    "notes_coverage_suffix": {
        "en": " Notes are currently loaded for chapters {covered}.",
        "bn": " বর্তমানে {covered} chapter-গুলোর জন্য notes লোড করা আছে।",
    },
}


def tr(key: str, lang: str, **kwargs) -> str:
    """Translated Python-generated message, formatted with kwargs.

    Falls back to English for an unknown language or a missing key,
    rather than raising -- a message that only degrades to English is
    far better than one that crashes the pipeline.
    """
    entry = MESSAGES.get(key)
    if not entry:
        return key
    template = entry.get(lang, entry["en"])
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError):
        return entry["en"].format(**kwargs)
