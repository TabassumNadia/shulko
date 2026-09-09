"""
Extractor agent -- invoice in, structured line items out.

This is the entry point of the whole pipeline, so its failures are the
expensive ones: a misread quantity propagates through classification and
duty calculation into a wrong number the user may act on.

Hence two rules in the prompt and enforced here: transcribe rather than
correct, and leave a field null rather than guess it.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.output_parsers import PydanticOutputParser
from langsmith import traceable
from pydantic import BaseModel, Field

from backend.core.config import get_settings
from backend.core.llm import get_llm, with_retry
from backend.core.schemas import ItemAttributes, LineItem
from backend.prompts.extract import EXTRACT_SYSTEM, EXTRACT_USER
from backend.tools.ocr_tool import image_blocks, read_document
from langchain_google_genai.chat_models import GoogleRateLimitError

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


class RawLineItem(BaseModel):
    """A flat shape for the vision call.

    LineItem nests ItemAttributes, which makes the extraction schema three
    levels deep. Vision models degrade sharply on nested schemas — they
    return an empty list rather than a malformed one, so the failure looks
    like an empty invoice. Flat here, nested afterwards in code.
    """
    description: str
    quantity: float = 1.0
    unit_price: float = 0.0
    currency: str = "USD"
    origin_country: str | None = None
    material: str | None = None
    form: str | None = None
    function: str | None = None


class RawExtraction(BaseModel):
    items: list[RawLineItem] = Field(default_factory=list)
    invoice_number: str | None = None
    supplier: str | None = None
    currency: str = "USD"


class ExtractedItems(BaseModel):
    items: list[LineItem] = Field(default_factory=list)
    invoice_number: str | None = None
    supplier: str | None = None
    currency: str = "USD"

    @classmethod
    def from_raw(cls, raw: RawExtraction) -> "ExtractedItems":
        return cls(
            invoice_number=raw.invoice_number,
            supplier=raw.supplier,
            currency=raw.currency,
            items=[
                LineItem(
                    description=r.description,
                    quantity=r.quantity,
                    unit_price=r.unit_price,
                    currency=r.currency or raw.currency,
                    origin_country=r.origin_country,
                    attributes=ItemAttributes(
                        material=r.material, form=r.form, function=r.function
                    ),
                )
                for r in raw.items
            ],
        )


@traceable(name="01_extract_line_items", run_type="chain")
def extract(file_path: str | Path) -> ExtractedItems:
    """Read an invoice and return typed line items with attributes."""
    path = Path(file_path)
    settings = get_settings()

    parser = PydanticOutputParser(pydantic_object=RawExtraction)
    # The vision call is the first thing an invoice run does, so a 429
    # here loses the whole run before any work is done. Retry it.
    model = with_retry(get_llm("vision"))

    if path.suffix.lower() in IMAGE_SUFFIXES:
        result = ExtractedItems.from_raw(_extract_from_image(model, parser, path))
    else:
        text = read_document(path)
        if not text.strip():
            raise ValueError(
                f"No text could be read from {path.name}. "
                "Try a clearer photo or a PDF with a text layer."
            )
        reply = model.invoke([
            ("system", EXTRACT_SYSTEM + "\n\n" + parser.get_format_instructions()),
            ("user", [{"type": "text", "text": f"{EXTRACT_USER}\n\n{text[:20000]}"}]),
        ])
        result = ExtractedItems.from_raw(_parse(parser, reply))

    # Cap the work so one 200-line invoice cannot run up a huge bill.
    result.items = [_clean(i, result.currency) for i in result.items][
        : settings.max_line_items
    ]
    return result


def _message_text(reply) -> str:
    """The reply text, across LangChain message versions."""
    text = getattr(reply, "text", None)
    if callable(text):
        text = text()
    return text or str(getattr(reply, "content", reply))


def _parse(parser: PydanticOutputParser, reply) -> RawExtraction:
    """Parse the model's JSON, tolerating a fenced code block around it."""

    text = _message_text(reply).strip()

    try:
        return parser.parse(text)

    except Exception:
        import json
        import re

        # Extract JSON object if the model wrapped it in text/code fences.
        match = re.search(r"\{.*\}", text, re.S)

        if not match:
            raise ValueError(
                "Model response did not contain a valid JSON object.\n\n"
                f"Raw response:\n{text}"
            )

        return RawExtraction.model_validate(
            json.loads(match.group(0))
        )


def _extract_from_image(model, parser: PydanticOutputParser,
                        path: Path) -> "RawExtraction":
    """Read an invoice image, asking for JSON directly.

    Deliberately NOT with_structured_output. That routes the request
    through the SDK's automatic function-calling loop, which on a
    multimodal call stalls with no error and no timeout — the single
    hardest failure in this project to diagnose, because a hang looks
    identical to a slow model.

    A plain call plus a parser is also easier to defend: the prompt says
    what shape to return, and PydanticOutputParser validates it.

    The image block format is still tried in order, because an
    unrecognised block is dropped silently and the model then answers
    from the text alone.
    """
    system = EXTRACT_SYSTEM + "\n\n" + parser.get_format_instructions()
    last_error = None

    for label, block in image_blocks(path):
        try:
            reply = model.invoke([
                ("system", system),
                ("user", [{"type": "text", "text": EXTRACT_USER}, block]),
            ])
            result = _parse(parser, reply)
        except GoogleRateLimitError as exc:
            raise ValueError(
               f"Gemini quota/rate limit reached while reading {path.name}. "
                "Please wait and try again."
            ) from exc

        except Exception as exc:
            last_error = exc
            print(f"[extractor] {label} image block failed: {exc}")
            continue

        if result and result.items:
            print(f"[extractor] read {len(result.items)} items "
                  f"using the {label} image block")
            return result
        print(f"[extractor] {label} image block returned no items, retrying")

    if last_error:
        raise ValueError(f"Could not read {path.name}: {last_error}") from last_error

    raise ValueError(
        f"{path.name} reached the model but produced no line items. "
        "If the document really is an invoice, the image block format may "
        "have changed again. Run with --raw to see what the model actually "
        "sees: python -m backend.agents.extractor <file> --raw"
    )


def _clean(item: LineItem, invoice_currency: str) -> LineItem:
    """Normalise without inventing. Blank attributes stay blank."""
    item.description = " ".join(item.description.split())
    if not item.currency:
        item.currency = invoice_currency
    if item.attributes is None:
        item.attributes = ItemAttributes()
    item.quantity = max(item.quantity, 0.0)
    item.unit_price = max(item.unit_price, 0.0)
    return item


def describe_raw(file_path: str | Path) -> str:
    """Ask the model to describe the image, with no schema in the way.

    The fastest way to answer the only question that matters when
    extraction returns nothing: did the model actually receive a picture?
    If this prints invoice text, the image is arriving and the problem is
    the schema. If it says it cannot see an image, the block format is.
    """
    from backend.tools.ocr_tool import image_blocks

    llm = get_llm("vision")
    for label, block in image_blocks(Path(file_path)):
        try:
            reply = llm.invoke([(
                "user",
                [{"type": "text",
                  "text": "What document is this? Quote its first three lines."},
                 block],
            )])
        except Exception as exc:
            print(f"[{label}] failed: {exc}")
            continue
        text = _message_text(reply)
        print(f"\n--- {label} block ---\n{text.strip()}\n")
        return text
    return ""


if __name__ == "__main__":
    # Test OCR on its own, without the API or the rest of the pipeline:
    #     python -m backend.agents.extractor data/samples/invoice_01_clean.png
    #     python -m backend.agents.extractor data/samples/invoice_01_clean.png --raw
    import json
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m backend.agents.extractor <invoice file> [--raw]")

    if "--raw" in sys.argv:
        describe_raw(sys.argv[1])
        raise SystemExit(0)

    extracted = extract(sys.argv[1])
    print(f"supplier: {extracted.supplier}")
    print(f"invoice:  {extracted.invoice_number}")
    print(f"currency: {extracted.currency}")
    print(f"items:    {len(extracted.items)}\n")
    for item in extracted.items:
        print(json.dumps(item.model_dump(), indent=2, ensure_ascii=False))
