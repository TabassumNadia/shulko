"""
Tariff ingestion: PDF -> structured files -> Chroma.

Two sources, two jobs:

  consolidated tariff PDF   -> rates      (CD, SD, VAT, AIT, RD, AT)
      customs.gov.bd/files/Tariff-2025-2026(29-07-2025).pdf

  per-chapter PDFs          -> notes      (exclusions, definitions)
      nbr.gov.bd/uploads/tariff_schedule/Chapter-39.pdf

The chapter PDFs only carry the statutory import duty, so they cannot
supply the full rate set. The consolidated schedule cannot supply the
chapter notes. You need both.

Usage
-----
    python -m backend.rag.ingest inspect data/raw/Tariff-2025-2026.pdf --page 40
    python -m backend.rag.ingest rates   data/raw/Tariff-2025-2026.pdf --chapters 39 61 62 84 85 87
    python -m backend.rag.ingest notes   data/raw/Chapter-39.pdf data/raw/Chapter-84.pdf
    python -m backend.rag.ingest index
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

PROCESSED = Path("data/processed")
TARIFF_CSV = PROCESSED / "tariff.csv"
NOTES_JSONL = PROCESSED / "notes.jsonl"

# 8-digit Bangladesh HS code: 3901.20.10
HS_RE = re.compile(r"\b(\d{4}\.\d{2}\.\d{2})\b")
# A percentage or bare number in a rate column
RATE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%?")

CSV_FIELDS = ["hs_code", "chapter", "heading", "description",
              "unit", "cd", "rd", "sd", "vat", "ait", "at"]

# Statutory defaults, applied when a row does not list its own value.
DEFAULTS = {"rd": 0.0, "sd": 0.0, "vat": 15.0, "ait": 5.0, "at": 5.0}


def _require_pdfplumber():
    try:
        import pdfplumber  # noqa: F401
        return __import__("pdfplumber")
    except ImportError:
        sys.exit("pdfplumber is not installed. Run: pip install pdfplumber")


# --------------------------------------------------------------- inspect
def cmd_inspect(args) -> None:
    """Print what pdfplumber actually sees, so the parser can be adapted.

    Run this FIRST on any new PDF. Paste the output into the chat if the
    parser below does not match the layout.
    """
    pdfplumber = _require_pdfplumber()
    with pdfplumber.open(args.pdf) as pdf:
        print(f"pages: {len(pdf.pages)}")
        page = pdf.pages[args.page]
        print(f"\n--- page {args.page} text (first 2000 chars) ---")
        print((page.extract_text() or "")[:2000])

        tables = page.extract_tables()
        print(f"\n--- page {args.page} tables: {len(tables)} ---")
        for t in tables[:1]:
            for row in t[:12]:
                print(row)


# --------------------------------------------------------------- rates
def _parse_rate_row(line: str) -> dict | None:
    """Pull one tariff row out of a line of text.

    Layout varies between editions, so this is deliberately forgiving:
    find the HS code, take the text after it as the description, and read
    the trailing numbers as the rate columns.
    """
    m = HS_RE.search(line)
    if not m:
        return None

    hs = m.group(1)
    tail = line[m.end():].strip()

    # Trailing numbers are the rate columns, in schedule order.
    numbers = RATE_RE.findall(tail)
    description = RATE_RE.sub("", tail).strip(" .|\t")

    row = {
        "hs_code": hs,
        "chapter": hs[:2],
        "heading": hs[:4],
        "description": re.sub(r"\s+", " ", description)[:400],
        "unit": "",
        **DEFAULTS,
        "cd": 0.0,
    }

    # Consolidated schedule order: CD SD VAT AIT RD AT (TTI ignored).
    order = ["cd", "sd", "vat", "ait", "rd", "at"]
    for key, value in zip(order, numbers):
        try:
            row[key] = float(value)
        except ValueError:
            pass

    return row if row["description"] else None


def cmd_rates(args) -> None:
    """Consolidated tariff PDF -> data/processed/tariff.csv"""
    pdfplumber = _require_pdfplumber()
    wanted = {str(c).zfill(2) for c in args.chapters} if args.chapters else None

    rows: dict[str, dict] = {}
    with pdfplumber.open(args.pdf) as pdf:
        for i, page in enumerate(pdf.pages):
            for line in (page.extract_text() or "").splitlines():
                row = _parse_rate_row(line)
                if not row:
                    continue
                if wanted and row["chapter"] not in wanted:
                    continue
                rows[row["hs_code"]] = row      # dedupe on HS code
            if i % 100 == 0:
                print(f"  page {i}… {len(rows)} rows", file=sys.stderr)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    with TARIFF_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows.values())

    print(f"wrote {len(rows)} rows -> {TARIFF_CSV}")
    if rows:
        print("\nSanity check — first 3 rows:")
        for r in list(rows.values())[:3]:
            print(f"  {r['hs_code']}  CD={r['cd']}%  SD={r['sd']}%  {r['description'][:60]}")
    print("\nOpen the CSV and eyeball it before indexing. Garbage in, garbage out.")


# --------------------------------------------------------------- notes
NOTE_START = re.compile(r"^\s*(\d{1,2})\s*[.\-]\s+(.*)")
TABLE_START = re.compile(r"H\.?\s?S\.?\s*Code|Heading\s+H", re.I)


def cmd_notes(args) -> None:
    """Per-chapter PDFs -> data/processed/notes.jsonl

    Notes sit before the tariff table. We read until the table header
    appears, then stop — everything after that is rates, not law.
    """
    pdfplumber = _require_pdfplumber()
    PROCESSED.mkdir(parents=True, exist_ok=True)

    out = []
    for path in args.pdfs:
        chapter = re.search(r"(\d{1,2})", Path(path).stem)
        chapter = chapter.group(1).zfill(2) if chapter else "??"

        with pdfplumber.open(path) as pdf:
            text = "\n".join((p.extract_text() or "") for p in pdf.pages[:4])

        current_no, buffer = None, []
        for line in text.splitlines():
            if TABLE_START.search(line):
                break
            m = NOTE_START.match(line)
            if m:
                if current_no and buffer:
                    out.append(_note(chapter, current_no, buffer))
                current_no, buffer = m.group(1), [m.group(2)]
            elif current_no and line.strip():
                buffer.append(line.strip())

        if current_no and buffer:
            out.append(_note(chapter, current_no, buffer))
        print(f"  chapter {chapter}: {sum(1 for n in out if n['chapter'] == chapter)} notes")

    with NOTES_JSONL.open("w", encoding="utf-8") as f:
        for note in out:
            f.write(json.dumps(note, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} notes -> {NOTES_JSONL}")


def _note(chapter: str, no: str, buffer: list[str]) -> dict:
    body = re.sub(r"\s+", " ", " ".join(buffer)).strip()
    lowered = body.lower()
    kind = ("exclusion" if "does not cover" in lowered or "excludes" in lowered
            else "definition" if "means" in lowered or "expression" in lowered
            else "general")
    return {
        "chapter": chapter,
        "note_no": no,
        "note_type": kind,
        "source": f"Chapter {chapter}, Note {no}",
        "text": body,
    }


# --------------------------------------------------------------- index
def cmd_index(args) -> None:
    """Load the processed files into the LangChain Chroma vector store.

    Resumable: rows already embedded are skipped, so an interrupted run
    picks up where it stopped instead of starting over.
    """
    from backend.core.config import get_settings
    from backend.rag.store import (
        NOTES_COLLECTION, TARIFF_COLLECTION, collection_size,
        index_chapter_notes, index_tariff_lines,
    )

    settings = get_settings()
    print(f"embedding provider: {settings.embedding_provider}")

    if TARIFF_CSV.exists():
        with TARIFF_CSV.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if args.chapters:
            wanted = {str(c).zfill(2) for c in args.chapters}
            rows = [r for r in rows if r["chapter"] in wanted]
            print(f"limited to chapters {sorted(wanted)}: {len(rows)} rows")

        print(f"embedding {len(rows)} tariff lines…")
        added = index_tariff_lines(rows, resume=not args.fresh)
        print(f"added {added} tariff lines "
              f"({collection_size(TARIFF_COLLECTION)} in the collection)")
    else:
        print(f"missing {TARIFF_CSV} — run the 'rates' command first")

    if NOTES_JSONL.exists():
        notes = [json.loads(l) for l in NOTES_JSONL.open(encoding="utf-8") if l.strip()]
        added = index_chapter_notes(notes, resume=not args.fresh)
        print(f"added {added} chapter notes "
              f"({collection_size(NOTES_COLLECTION)} in the collection)")
    else:
        print(f"missing {NOTES_JSONL} — run the 'notes' command first")


# --------------------------------------------------------------- cli
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("inspect", help="see what pdfplumber finds on a page")
    i.add_argument("pdf")
    i.add_argument("--page", type=int, default=0)
    i.set_defaults(func=cmd_inspect)

    r = sub.add_parser("rates", help="consolidated tariff PDF -> tariff.csv")
    r.add_argument("pdf")
    r.add_argument("--chapters", nargs="*", default=None,
                   help="chapter numbers to keep, e.g. 39 61 84")
    r.set_defaults(func=cmd_rates)

    n = sub.add_parser("notes", help="per-chapter PDFs -> notes.jsonl")
    n.add_argument("pdfs", nargs="+")
    n.set_defaults(func=cmd_notes)

    x = sub.add_parser("index", help="load processed files into Chroma")
    x.add_argument("--chapters", nargs="*", default=None,
                   help="index only these chapters, e.g. 39 61 84")
    x.add_argument("--fresh", action="store_true",
                   help="re-embed everything instead of resuming")
    x.set_defaults(func=cmd_index)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
