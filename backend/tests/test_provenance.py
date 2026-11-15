"""Citation provenance: a hash a reader can recompute.

The product claim is that every sentence is backed by a passage. The hash
makes the second half of that claim checkable without trusting the server:
sha256 over the document checksum, the character span and the text. A
client that holds those four things can prove the quote is that exact span
of that exact file.
"""

from __future__ import annotations

import hashlib

from app.models.schemas import ClassificationLevel
from app.rag.chunking import chunk_text, provenance_hash, simple_word_offsets

DOC = "# Policy\n\nRefunds take 30 days. " + ("Contact support with your order id. " * 40)


def test_hash_is_stable_across_re_ingestion() -> None:
    a = chunk_text("d", DOC, simple_word_offsets, 20, 5, document_checksum="abc")
    b = chunk_text("d", DOC, simple_word_offsets, 20, 5, document_checksum="abc")
    assert [c.provenance_hash for c in a] == [c.provenance_hash for c in b]
    assert all(len(c.provenance_hash) == 64 for c in a)


def test_hash_changes_when_the_source_file_changes() -> None:
    a = chunk_text("d", DOC, simple_word_offsets, 20, 5, document_checksum="abc")
    b = chunk_text("d", DOC, simple_word_offsets, 20, 5, document_checksum="abd")
    assert all(x.provenance_hash != y.provenance_hash for x, y in zip(a, b, strict=True))


def test_hash_matches_the_published_recipe() -> None:
    """The recipe in ChunkDetail.hash_recipe must be exactly this, so a
    client in any language can verify without reading server code."""
    chunks = chunk_text("d", DOC, simple_word_offsets, 20, 5, document_checksum="abc")
    for c in chunks:
        expected = hashlib.sha256(f"abc|{c.char_start}|{c.char_end}|{c.text}".encode()).hexdigest()
        assert (
            c.provenance_hash
            == expected
            == provenance_hash("abc", c.char_start, c.char_end, c.text)
        )


def test_access_metadata_is_stamped_on_every_chunk() -> None:
    chunks = chunk_text(
        "d",
        DOC,
        simple_word_offsets,
        20,
        5,
        tenant_id="acme",
        allowed_roles=["finance"],
        classification=ClassificationLevel.CONFIDENTIAL,
    )
    assert all(c.tenant_id == "acme" for c in chunks)
    assert all(c.allowed_roles == ["finance"] for c in chunks)
    assert all(c.classification == ClassificationLevel.CONFIDENTIAL for c in chunks)


def test_a_window_spanning_a_page_break_is_attributed_to_its_own_page() -> None:
    """A block that runs across a page boundary used to stamp every window
    with the page it started on, so text on page 2 was cited as page 1."""
    from app.rag.pdf import PAGE_MARKER_PREFIX, PAGE_MARKER_SUFFIX

    page1 = "word " * 30
    page2 = "other " * 30
    text = (
        f"{PAGE_MARKER_PREFIX}1{PAGE_MARKER_SUFFIX}\n{page1}"
        f"{PAGE_MARKER_PREFIX}2{PAGE_MARKER_SUFFIX}\n{page2}"
    )
    chunks = chunk_text("d", text, simple_word_offsets, 10, 0)
    pages = {c.page_number for c in chunks}
    assert pages == {1, 2}
    for c in chunks:
        if c.page_number == 2:
            assert "other" in c.text


def test_the_chunk_endpoint_returns_a_hash_a_client_can_verify(api, offline_env) -> None:
    """End to end, the way the browser does it: fetch the chunk, recompute
    sha256 from the published recipe, compare. Regression: shared-corpus
    chunks have no registry row, the endpoint returned an empty checksum,
    and every corpus citation showed as a provenance MISMATCH."""
    from .conftest import publish_chunks

    chunks = chunk_text("refund-policy", DOC, simple_word_offsets, 20, 5, document_checksum="f00d")
    publish_chunks(chunks)
    c = chunks[0]
    url = f"/v1/documents/refund-policy/chunks/{c.chunk_id}"
    body = api.get(url, headers={"x-api-key": "contractor-local"}).json()
    parts = ("document_checksum_sha256", "char_start", "char_end", "text")
    payload = "|".join(str(body[k]) for k in parts)
    assert body["document_checksum_sha256"] == "f00d"
    assert hashlib.sha256(payload.encode()).hexdigest() == body["provenance_hash"]
