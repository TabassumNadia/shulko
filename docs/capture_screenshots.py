"""
Capture the README screenshots from the running app.

Kept in the repository because a screenshot goes stale the moment the UI
changes, and a script that regenerates it is worth more than a PNG nobody
can reproduce.

    # terminal 1
    uvicorn backend.main:app
    # terminal 2
    streamlit run frontend/app.py
    # terminal 3
    python docs/capture_screenshots.py

Requires playwright, which is a development dependency only:

    pip install playwright && playwright install chromium
"""

from __future__ import annotations

import sys
from pathlib import Path

APP = "http://localhost:8501"
OUT = Path(__file__).parent / "screenshots"

# Wide enough that the sidebar and the report sit side by side, which is
# how the app is actually used.
VIEWPORT = {"width": 1440, "height": 1000}

# Streamlit re-runs the whole script on every interaction, so "the page
# has loaded" is not the same as "the answer has arrived". Each step
# waits for text that only appears once that step is really done.
QUESTION = "what is the import duty on cotton t-shirts?"


def _settle(page, timeout: int = 60_000) -> None:
    """Wait until Streamlit stops running the script."""
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')",
        timeout=timeout,
    )


def _wait_for_answer(page, timeout: int = 420_000) -> None:
    """Wait for the report itself, not merely for the page to go quiet.

    The pipeline runs behind an in-message spinner rather than Streamlit's
    status widget, so the script looks idle while the answer is still
    being computed -- which is how the first version of this captured a
    screenshot of the word "Running the pipeline...".

    A rate card is the answer to a typed question, and its "Total tax
    incidence" metric is the last thing rendered; the wait ends there.
    """
    page.wait_for_selector("text=Total tax incidence", timeout=timeout)


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed. "
              "Run: pip install playwright && playwright install chromium")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)

        page.goto(APP, wait_until="networkidle", timeout=60_000)
        _settle(page)
        page.wait_for_timeout(1500)
        page.screenshot(path=OUT / "01_start.png")
        print("wrote 01_start.png")

        # The question path: no upload, straight to a rate card.
        page.get_by_placeholder("Ask about a tariff code…").fill(QUESTION)
        page.keyboard.press("Enter")
        _wait_for_answer(page)
        page.wait_for_timeout(1500)
        page.screenshot(path=OUT / "02_question.png", full_page=True)
        print("wrote 02_question.png")

        # The reasoning behind the code, which is the part worth showing:
        # the citation, not just the number.
        why = page.get_by_text("Why this HS code?").first
        if why.count():
            why.click()
            page.wait_for_timeout(800)
            page.screenshot(path=OUT / "03_report.png", full_page=True)
            print("wrote 03_report.png")

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
