"""Chunker unit tests — fully offline via the injected word tokenizer."""

from app.rag.chunking import chunk_text, simple_word_offsets

DOC = """# Refund Policy

## Eligibility
Customers may request a refund within 30 days of purchase. The product must be
unused and in its original packaging. Digital goods are refundable only if the
license key was never activated.

## Fees

| Region | Restocking fee | Processing days |
|--------|----------------|-----------------|
| US     | 5%             | 3               |
| EU     | 0%             | 5               |

Fees are waived for defective items.

## Escalation
""" + ("Contact support with your order id. " * 80)  # long prose to force windowing


def _chunks(size: int = 50, overlap: int = 10):
    return chunk_text("policy", DOC, simple_word_offsets, size, overlap)


def test_deterministic_ids() -> None:
    a, b = _chunks(), _chunks()
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    # ids are namespaced by the caller-supplied document id, never a filename
    assert all(c.chunk_id.startswith("policy_") for c in a)
    assert all(c.doc_id == "policy" for c in a)


def test_chunk_ids_are_namespaced_by_document_id() -> None:
    """Two documents with identical content must not share chunk ids.

    This is the regression guard for the old filename-derived identity: two
    users uploading the same file produced colliding ids that overwrote each
    other in the index.
    """
    a = chunk_text("doc-aaa", DOC, simple_word_offsets, 50, 10)
    b = chunk_text("doc-bbb", DOC, simple_word_offsets, 50, 10)
    assert {c.chunk_id for c in a}.isdisjoint({c.chunk_id for c in b})


def test_owner_and_filename_are_carried_onto_chunks() -> None:
    chunks = chunk_text(
        "doc-1", DOC, simple_word_offsets, 50, 10, owner_id="user-7", source_filename="p.pdf"
    )
    assert all(c.owner_id == "user-7" for c in chunks)
    assert all(c.source_filename == "p.pdf" for c in chunks)


def test_max_chunks_is_enforced() -> None:
    capped = chunk_text("doc-1", DOC, simple_word_offsets, 50, 10, max_chunks=3)
    assert len(capped) == 3


def test_table_is_atomic() -> None:
    tables = [c for c in _chunks(size=5, overlap=1) if c.is_table]  # tiny window: would split prose
    assert len(tables) == 1
    t = tables[0]
    assert "Restocking fee" in t.text and "| EU" in t.text  # header and last row together
    assert t.section_path == "Refund Policy > Fees"


def test_prose_windows_overlap() -> None:
    esc = [c for c in _chunks(size=50, overlap=10) if c.section_path.endswith("Escalation")]
    assert len(esc) >= 2, "long section must produce multiple windows"
    # consecutive windows share text (the 10-token overlap)
    tail = " ".join(esc[0].text.split()[-5:])
    assert tail in esc[1].text


def test_section_paths_nest() -> None:
    paths = {c.section_path for c in _chunks()}
    assert "Refund Policy > Eligibility" in paths
    assert "Refund Policy > Escalation" in paths


def test_offsets_reconstruct_text() -> None:
    for c in _chunks():
        assert c.text == DOC[c.char_start : c.char_end].strip()
