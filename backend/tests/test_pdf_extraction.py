"""PDF extraction against a deterministic, programmatically-built corpus.

The point of these tests is that the pipeline must either extract the document
FAITHFULLY or FAIL LOUDLY — the one outcome that must never happen is silently
indexing an unreadable document as an empty success, because the user then gets
confident refusals about a document they believe was ingested.
"""

import pytest
from app.config import get_settings
from app.models.schemas import DocumentErrorCode
from app.rag import pdf as pdf_mod
from app.rag.chunking import chunk_text, simple_word_offsets
from app.rag.pdf import PdfExtractionError, blocks_to_markdown, extract_pdf

from . import pdf_fixtures as fx


@pytest.fixture(autouse=True)
def _clear_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------- happy paths


def test_simple_pdf_text_is_extracted(tmp_path) -> None:
    doc = extract_pdf(fx.simple_text_pdf(tmp_path / "simple.pdf"))
    text = blocks_to_markdown(doc)
    assert fx.FACT_PAGE_1 in text
    assert doc.page_count == 1


def test_multipage_pdf_tags_every_page(tmp_path) -> None:
    doc = extract_pdf(fx.multipage_pdf(tmp_path / "multi.pdf"))
    assert doc.page_count == 3
    pages = {b.page_number for b in doc.blocks}
    assert pages == {1, 2, 3}


def test_facts_are_attributed_to_the_correct_page(tmp_path) -> None:
    """Citations claim a page number; that number has to be right."""
    doc = extract_pdf(fx.multipage_pdf(tmp_path / "multi.pdf"))

    def page_of(fact: str) -> int | None:
        for block in doc.blocks:
            if fact.split(".")[0][:40] in block.text:
                return block.page_number
        return None

    assert page_of(fx.FACT_PAGE_1) == 1
    assert page_of(fx.FACT_PAGE_2) == 2
    assert page_of(fx.FACT_PAGE_3) == 3


def test_running_header_is_not_repeated_into_every_chunk(tmp_path) -> None:
    """A header repeated on every page would otherwise dominate retrieval."""
    doc = extract_pdf(fx.multipage_pdf(tmp_path / "multi.pdf"))
    header_hits = sum(
        1 for b in doc.blocks if b.text.strip() == "Meridian Robotics - Internal Handbook"
    )
    assert header_hits == 0


def test_page_number_footers_are_dropped(tmp_path) -> None:
    doc = extract_pdf(fx.multipage_pdf(tmp_path / "multi.pdf"))
    assert not any(b.text.strip().isdigit() for b in doc.blocks)


def test_table_rows_survive_extraction(tmp_path) -> None:
    """A table question ("what does a Designer get?") needs the row intact."""
    doc = extract_pdf(fx.multipage_pdf(tmp_path / "multi.pdf"))
    text = blocks_to_markdown(doc)
    assert fx.TABLE_FACT_ROLE in text
    assert fx.TABLE_FACT_BUDGET in text


def test_two_column_pdf_keeps_columns_separate(tmp_path) -> None:
    """Reading across the fold splices unrelated sentences together."""
    doc = extract_pdf(fx.two_column_pdf(tmp_path / "cols.pdf"))
    text = blocks_to_markdown(doc)
    left_pos = text.find("shipping timelines")
    right_pos = text.find("warranty terms")
    assert left_pos != -1 and right_pos != -1
    # the whole left column must be emitted before the right one starts
    assert text.find("continues for several lines") < right_pos


def test_unicode_and_numeric_content_survives(tmp_path) -> None:
    doc = extract_pdf(fx.unicode_pdf(tmp_path / "uni.pdf"))
    text = blocks_to_markdown(doc)
    assert "1.234,56" in text
    assert "5%" in text


# ---------------------------------------------------------------- failure paths


def test_encrypted_pdf_is_rejected_not_silently_empty(tmp_path) -> None:
    with pytest.raises(PdfExtractionError) as exc:
        extract_pdf(fx.encrypted_pdf(tmp_path / "enc.pdf"))
    assert exc.value.code == DocumentErrorCode.ENCRYPTED_DOCUMENT


def test_image_only_pdf_reports_scanned_rather_than_indexing_nothing(tmp_path) -> None:
    with pytest.raises(PdfExtractionError) as exc:
        extract_pdf(fx.empty_pdf(tmp_path / "blank.pdf"))
    assert exc.value.code == DocumentErrorCode.UNSUPPORTED_SCANNED_DOCUMENT
    assert "scanned" in exc.value.message.lower()


def test_corrupt_pdf_is_rejected(tmp_path) -> None:
    with pytest.raises(PdfExtractionError) as exc:
        extract_pdf(fx.corrupt_pdf(tmp_path / "bad.pdf"))
    assert exc.value.code == DocumentErrorCode.CORRUPT_DOCUMENT


def test_page_limit_is_enforced(tmp_path, monkeypatch) -> None:
    """A 1 MB file can still hold hundreds of pages; size is not a work bound."""
    monkeypatch.setenv("MAX_PDF_PAGES", "2")
    get_settings.cache_clear()
    with pytest.raises(PdfExtractionError) as exc:
        extract_pdf(fx.multipage_pdf(tmp_path / "multi.pdf"))
    assert exc.value.code == DocumentErrorCode.TOO_MANY_PAGES


# ---------------------------------------------------------------- chunk wiring


def test_chunks_carry_page_numbers_through_to_retrieval(tmp_path) -> None:
    doc = extract_pdf(fx.multipage_pdf(tmp_path / "multi.pdf"))
    chunks = chunk_text("doc-1", blocks_to_markdown(doc), simple_word_offsets, 120, 20)
    assert chunks, "expected chunks from a 3-page document"
    assert any(c.page_number == 1 for c in chunks)
    assert any(c.page_number == 3 for c in chunks)


def test_page_markers_never_leak_into_chunk_text(tmp_path) -> None:
    """Markers are plumbing; they must not reach embeddings, the LLM, or quotes."""
    doc = extract_pdf(fx.multipage_pdf(tmp_path / "multi.pdf"))
    chunks = chunk_text("doc-1", blocks_to_markdown(doc), simple_word_offsets, 120, 20)
    assert all(pdf_mod.PAGE_MARKER_PREFIX not in c.text for c in chunks)
    assert all(c.text.strip() for c in chunks)
