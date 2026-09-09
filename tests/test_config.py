"""
Tests for configuration and LangSmith wiring.

The failure these guard against is the quiet one: a key sitting in .env
that never reaches os.environ, so the app runs fine and produces no
traces at all — discovered the night before submission.
"""

import os

from backend.core.config import Settings, configure_tracing


def test_new_style_langsmith_variables_are_read():
    s = Settings(langsmith_api_key="lsv2_test", langsmith_project="shulko")
    assert s.tracing_api_key == "lsv2_test"
    assert s.tracing_project == "shulko"
    assert s.tracing_enabled


def test_legacy_langchain_variables_still_work():
    """A .env copied from older docs must not silently disable tracing.

    Every new-style field is passed explicitly as empty, because that is
    the condition under test: legacy names are used only when the current
    ones are absent. Leaving `langsmith_project` unset would let the
    ambient environment answer instead of the test -- and it does, since
    importing the package publishes a resolved project name into
    os.environ for the tracers to read.
    """
    s = Settings(
        langsmith_api_key="", langsmith_project="", langsmith_tracing=False,
        langchain_api_key="lsv2_legacy", langchain_project="old_name",
        langchain_tracing_v2=True,
    )
    assert s.tracing_api_key == "lsv2_legacy"
    assert s.tracing_project == "old_name"
    assert s.tracing_enabled


def test_new_style_wins_when_both_are_present():
    s = Settings(langsmith_api_key="new", langchain_api_key="old")
    assert s.tracing_api_key == "new"


def test_tracing_is_off_without_a_key():
    s = Settings(langsmith_api_key="", langchain_api_key="",
                 langsmith_tracing=True)
    assert not s.tracing_enabled


def test_configure_tracing_publishes_both_spellings_to_environ():
    """The LangSmith client reads os.environ, not our Settings object."""
    before = {k: os.environ.get(k) for k in
              ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY",
               "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2",
               "LANGSMITH_PROJECT", "LANGCHAIN_PROJECT")}
    try:
        assert configure_tracing(
            Settings(langsmith_api_key="lsv2_env", langsmith_project="shulko")
        )
        assert os.environ["LANGSMITH_API_KEY"] == "lsv2_env"
        assert os.environ["LANGCHAIN_API_KEY"] == "lsv2_env"
        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGSMITH_PROJECT"] == "shulko"
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_configure_tracing_disables_explicitly_when_no_key():
    """An empty key must turn tracing off, not inherit a stale shell value."""
    os.environ["LANGSMITH_TRACING"] = "true"
    assert not configure_tracing(Settings(langsmith_api_key="",
                                          langchain_api_key=""))
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_defaults_are_sane():
    s = Settings()
    assert s.tracing_project == "shulko"
    assert 0 < s.confidence_threshold < 1
    assert s.max_line_items > 0


def test_every_model_id_is_configurable():
    """Model ids must not be literals in the code.

    Google retired text-embedding-004 and gemini-2.0-flash mid-project.
    Each retirement should cost a .env edit, not a code change.
    """
    s = Settings(
        chat_model="gemini-9-flash",
        vision_model="gemini-9-vision",
        judge_model="gemini-9-lite",
        embedding_model="models/embed-9",
        grounding_model="gemini-9-search",
    )
    assert s.chat_model == "gemini-9-flash"
    assert s.vision_model == "gemini-9-vision"
    assert s.judge_model == "gemini-9-lite"
    assert s.embedding_model == "models/embed-9"
    assert s.grounding_model == "gemini-9-search"


def test_no_retired_model_ids_are_used_as_values():
    """Guard against a retired id creeping back in as an actual value.

    Prose mentioning a retirement is fine and useful; a quoted string
    passed to a model constructor is what breaks in production.
    """
    from pathlib import Path

    retired = ["gemini-2.0-flash", "text-embedding-004", "gemini-1.5-flash"]
    offenders = []
    for path in Path("backend").rglob("*.py"):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for dead in retired:
                if f'"{dead}"' in line or f"'{dead}'" in line:
                    offenders.append(f"{path}:{lineno}")
    assert not offenders, (
        "retired model ids used as values in: " + ", ".join(offenders)
    )


def test_llm_roles_map_to_settings():
    from backend.core.llm import TEMPERATURES

    assert set(TEMPERATURES) == {"vision", "reasoning", "judge"}
    assert TEMPERATURES["judge"] == 0.0, "verification must be deterministic"


def test_embeddings_default_to_local():
    """Bulk-embedding a static price list must not compete with the
    agents for the Gemini free-tier quota."""
    s = Settings()
    assert s.embedding_provider == "fastembed"


def test_embedding_provider_is_switchable():
    s = Settings(embedding_provider="google")
    assert s.embedding_provider == "google"


def test_unknown_embedding_provider_fails_with_a_useful_message():
    import os
    from backend.rag import store

    original = store.get_settings
    store.get_settings = lambda: Settings(embedding_provider="nonsense")
    try:
        try:
            store.get_embeddings()
            raise AssertionError("should have raised")
        except RuntimeError as exc:
            assert "fastembed" in str(exc) and "google" in str(exc)
    finally:
        store.get_settings = original


def test_lite_models_are_not_sent_a_temperature():
    """Fixed-sampling models warn on every call if given one."""
    import inspect
    from backend.core import llm

    source = inspect.getsource(llm.get_llm)
    assert '"-lite"' in source and '"-preview"' in source


def test_collections_are_namespaced_by_embedding_provider():
    """A Chroma collection fixes its vector dimensionality on creation.

    Gemini embeddings are 768-dim, bge-small is 384. Sharing one
    collection name across providers fails mid-index with a dimension
    mismatch, so the provider belongs in the name.
    """
    from backend.rag import store

    original = store.get_settings
    try:
        store.get_settings = lambda: Settings(embedding_provider="fastembed")
        local = store.collection_name("tariff_lines")

        store.get_settings = lambda: Settings(embedding_provider="google")
        remote = store.collection_name("tariff_lines")
    finally:
        store.get_settings = original

    assert local != remote
    assert "fastembed" in local and "google" in remote


def test_note_ids_survive_duplicate_numbering():
    """Chapters number "Notes" and "Subheading Notes" separately, so both
    contain a Note 1. Chroma rejects an entire batch on one duplicate id."""
    from backend.rag.store import _unique_ids

    notes = [
        {"chapter": "39", "scope": "chapter", "note_no": "1"},
        {"chapter": "39", "scope": "subheading", "note_no": "1"},
        {"chapter": "39", "scope": "chapter", "note_no": "2"},
    ]
    ids = _unique_ids(notes)
    assert len(set(ids)) == len(ids) == 3


def test_note_ids_degrade_rather_than_collide_on_bad_data():
    """A parsing quirk should cost one id, not the whole index."""
    from backend.rag.store import _unique_ids

    notes = [{"chapter": "84", "scope": "chapter", "note_no": "1"}] * 3
    ids = _unique_ids(notes)
    assert len(set(ids)) == 3


def test_parallel_preserves_order_and_runs_every_job():
    """Results must line up with their inputs — a per-item report attached
    to the wrong item would be worse than a slow one."""
    from backend.graph.builder import _parallel

    jobs = list(range(12))
    assert _parallel(lambda n: n * n, jobs) == [n * n for n in jobs]


def test_parallel_handles_the_single_item_case():
    from backend.graph.builder import _parallel

    assert _parallel(lambda n: n + 1, [41]) == [42]
    assert _parallel(lambda n: n + 1, []) == []


def test_parallel_copies_the_tracing_context():
    """LangSmith carries the parent run in a context variable. A bare
    thread starts outside it, and each item becomes its own root trace."""
    import inspect
    from backend.graph import builder

    source = inspect.getsource(builder._parallel)
    assert "copy_context" in source


def test_regulatory_checks_are_capped():
    s = Settings()
    assert 1 <= s.max_regulatory_checks <= 4


def test_image_blocks_are_the_shape_gemini_accepts():
    """An image block LangChain does not recognise is dropped silently,
    and the model then answers from the prompt text alone -- which looks
    like a bad extraction, not a bad block. So the shape is pinned here.

    The extractor loops over whatever this returns, so extra fallback
    formats can be added without touching it; what must not change is
    that the first block is the confirmed-working `v1` one.
    """
    import base64
    from pathlib import Path
    from backend.tools.ocr_tool import image_blocks

    sample = Path("data/samples/invoice_01_clean.png")
    if not sample.exists():
        import pytest
        pytest.skip("sample invoice not present")

    blocks = image_blocks(sample)
    assert blocks, "at least one image block must be offered"
    assert blocks[0][0] == "v1"

    v1 = blocks[0][1]
    assert v1["type"] == "image"
    assert "base64" in v1 and "mime_type" in v1
    base64.b64decode(v1["base64"])          # must be valid base64


def test_prepare_image_downscales_large_invoices():
    from pathlib import Path
    from backend.tools.ocr_tool import MAX_EDGE, prepare_image

    sample = Path("data/samples/invoice_01_clean.png")
    if not sample.exists():
        import pytest
        pytest.skip("sample invoice not present")

    payload, mime = prepare_image(sample)
    assert mime == "image/jpeg"
    assert len(payload) < sample.stat().st_size

    import io
    from PIL import Image
    assert max(Image.open(io.BytesIO(payload)).size) <= MAX_EDGE


def test_extraction_json_is_parsed_even_inside_a_code_fence():
    """Models wrap JSON in ```json fences. Losing an otherwise perfect
    extraction to three backticks would be an expensive way to fail."""
    from langchain_core.output_parsers import PydanticOutputParser
    from backend.agents.extractor import RawExtraction, _parse

    class _Reply:
        content = ('Here is the data:\n```json\n'
                   '{"items": [{"description": "Polypropylene granules", '
                   '"quantity": 12000, "unit_price": 1.28}], '
                   '"currency": "USD"}\n```')

    parser = PydanticOutputParser(pydantic_object=RawExtraction)
    result = _parse(parser, _Reply())

    assert len(result.items) == 1
    assert result.items[0].quantity == 12000


def test_flat_extraction_maps_into_nested_line_items():
    from backend.agents.extractor import ExtractedItems, RawExtraction, RawLineItem

    raw = RawExtraction(
        items=[RawLineItem(description="PP granules", quantity=100,
                           unit_price=2.5, material="polypropylene",
                           form="granules")],
        currency="USD",
    )
    out = ExtractedItems.from_raw(raw)

    assert out.items[0].attributes.material == "polypropylene"
    assert out.items[0].attributes.form == "granules"
    assert out.items[0].currency == "USD"


def test_vision_path_does_not_use_with_structured_output():
    """with_structured_output drives the SDK's function-calling loop,
    which stalls on multimodal calls with no error and no timeout."""
    import inspect
    from backend.agents import extractor

    # The docstring names it to explain why it is avoided, so look for
    # the call itself rather than the word.
    source = inspect.getsource(extractor)
    assert ".with_structured_output(" not in source
    assert "PydanticOutputParser" in source


def test_importing_the_backend_package_configures_tracing():
    """Tracing must not depend on which module got imported first.

    LangChain and LangSmith read os.environ when a run starts, and only
    `get_settings()` publishes our .env keys there. Before this was done
    eagerly in backend/__init__.py, running the graph directly produced
    no traces while running it behind the API produced full ones --
    same code, same key, different entry point, and no error either way.

    Importing the package is the one thing every entry point does, so
    that is where the guarantee lives.
    """
    import os
    import subprocess
    import sys

    # A fresh interpreter that imports nothing but the package, so an
    # already-configured parent process cannot mask the failure.
    result = subprocess.run(
        [sys.executable, "-c",
         "import backend, os; print(os.getenv('LANGSMITH_TRACING'))"],
        capture_output=True, text=True, cwd=os.getcwd(), timeout=120,
    )
    assert result.returncode == 0, result.stderr
    # "true" when a key is configured, "false" when none is -- either way
    # the variable is set deliberately rather than left to the shell.
    assert result.stdout.strip() in {"true", "false"}
