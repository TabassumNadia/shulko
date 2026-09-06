"""The single dict that travels through every node of the graph."""

from typing import TypedDict, Annotated
import operator


class ShulkoState(TypedDict, total=False):
    # --- inputs ---
    raw_file_path: str
    freight: float
    insurance: float
    importer_type: str
    language: str
    question: str            # set when the user asks instead of uploading

    # --- routing ---
    route: str               # "invoice" | "question"

    # --- produced by nodes ---
    ocr_text: str
    line_items: list[dict]
    results: Annotated[list[dict], operator.add]
    advisories: list[dict]
    needs_review: bool
    warnings: Annotated[list[str], operator.add]

    # --- output ---
    report: dict
