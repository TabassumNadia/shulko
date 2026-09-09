"""
OCR tool -- invoice image or PDF to text.

Three paths, in the order they are tried:

  PyMuPDF      For a PDF with a text layer. Pure-Python wheel, no system
               binary, and far faster than the alternatives. Most supplier
               invoices arrive as generated PDFs, so this handles the
               common case at almost no cost.

  RapidOCR     For images and scanned PDFs. An ONNX model of roughly
               15 MB that pip installs on Windows with no Tesseract
               binary, no PATH surgery, and no torch. This is the
               lightweight OCR model path.

  vision LLM   The Extractor's own route: Gemini reads the image and
               returns structured fields directly. On phone photos with
               stamps, skew and mixed fonts it beats classical OCR,
               because it understands layout rather than guessing at it
               (see arXiv 2603.02789, "OCR or Not?").

The tool returns raw text. Turning text into typed line items is the
Extractor agent's job, not this tool's -- keeping that boundary is what
lets either OCR path feed the same downstream pipeline.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from langsmith import traceable

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


@traceable(name="tool_ocr_read_document", run_type="tool")
def read_document(path: str | Path) -> str:
    """Extract raw text from an invoice file.

    Raises:
        FileNotFoundError: if the path does not exist.
        RuntimeError: if no extraction path is available, with a message
            naming the package to install.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return _ocr_image(path)
    raise RuntimeError(f"Unsupported file type: {path.suffix}")


def _read_pdf(path: Path) -> str:
    """Text layer first via PyMuPDF; OCR the rendered pages only if empty."""
    try:
        # `pymupdf` is the current module name; `fitz` is the legacy
        # alias, still shipped but deprecation-warned on newer releases.
        try:
            import pymupdf
        except ImportError:
            import fitz as pymupdf
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is not installed. Run: pip install pymupdf"
        ) from exc

    with pymupdf.open(path) as doc:
        text = "\n".join(page.get_text() for page in doc)
        if text.strip():
            return text

        # No text layer, so this is a scan. Render each page and OCR it.
        print(f"[ocr] {path.name} has no text layer, running OCR on {len(doc)} page(s)")
        pages = []
        for page in doc:
            pixmap = page.get_pixmap(dpi=200)
            pages.append(_ocr_bytes(pixmap.tobytes("png")))
    return "\n".join(pages)


def _ocr_image(path: Path) -> str:
    return _ocr_bytes(path.read_bytes())


def _ocr_bytes(data: bytes) -> str:
    """Run the lightweight ONNX OCR model over image bytes."""
    try:
        import numpy as np
        from PIL import Image
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "RapidOCR is not installed. Run: pip install rapidocr-onnxruntime"
        ) from exc

    import io

    engine = _engine()
    image = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    result, _ = engine(image)
    if not result:
        return ""
    # RapidOCR returns [box, text, confidence] per detected line.
    return "\n".join(line[1] for line in result)


_cached_engine = None


def _engine():
    """One engine instance for the process; loading the model is the slow part."""
    global _cached_engine
    if _cached_engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _cached_engine = RapidOCR()
    return _cached_engine


MAX_EDGE = 1600        # invoices are legible well below full resolution
JPEG_QUALITY = 82


def prepare_image(path: str | Path) -> tuple[bytes, str]:
    """Downscale and re-encode an invoice for the vision model.

    A 1800x2400 PNG is several hundred kilobytes of base64 on every call,
    and the extra pixels buy nothing: invoice text is legible at 1600px on
    the long edge. Shrinking it cuts both the upload and the model's
    processing time, which is most of the wait the user feels.
    """
    path = Path(path)
    raw = path.read_bytes()
    try:
        import io

        from PIL import Image

        image = Image.open(io.BytesIO(raw))
        image = image.convert("RGB")
        if max(image.size) > MAX_EDGE:
            ratio = MAX_EDGE / max(image.size)
            image = image.resize(
                (int(image.width * ratio), int(image.height * ratio)),
                Image.LANCZOS,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
        return buffer.getvalue(), "image/jpeg"
    except Exception:
        # Pillow missing or an odd format: send the original rather than fail.
        return raw, mimetypes.guess_type(path.name)[0] or "image/jpeg"


def image_blocks(path: str | Path) -> list[tuple[str, dict]]:
    """Create the confirmed working image block for Gemini Vision."""

    payload, mime = prepare_image(path)
    encoded = base64.b64encode(payload).decode()

    return [
        (
            "v1",
            {
                "type": "image",
                "base64": encoded,
                "mime_type": mime,
            },
        )
    ]


def as_data_part(path: str | Path) -> dict:
    """The current standard block. Kept for callers that want just one."""
    return image_blocks(path)[0][1]
