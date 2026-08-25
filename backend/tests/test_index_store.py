"""Index versioning, atomic publication, and concurrency safety.

These cover the two ways the previous single-directory, read-modify-write
layout lost data: torn publication (new vectors next to old chunk metadata) and
lost updates (two concurrent uploads, one silently erased).
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
from app.config import get_settings
from app.models.schemas import Chunk
from app.rag import bm25_store, index_store


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _chunk(cid: str, doc: str, text: str = "some text here") -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=doc,
        section_path="S",
        text=text,
        token_count=3,
        char_start=0,
        char_end=len(text),
    )


def _publish(tmp_path, chunks: list[Chunk], base: str | None) -> str:
    vecs = (
        np.random.rand(len(chunks), 8).astype(np.float32)
        if chunks
        else np.zeros((0, 8), np.float32)
    )
    index = index_store.build_faiss(vecs)
    bm25 = bm25_store.build([c.embed_text for c in chunks])
    return index_store.publish(index, chunks, bm25, base_version=base, root=tmp_path)


def test_first_publish_creates_v1_and_pointer(tmp_path) -> None:
    assert index_store.read_pointer(tmp_path) is None
    version = _publish(tmp_path, [_chunk("c1", "d1")], None)
    assert version == "v1"
    assert index_store.read_pointer(tmp_path) == "v1"


def test_versions_are_immutable_and_increment(tmp_path) -> None:
    v1 = _publish(tmp_path, [_chunk("c1", "d1")], None)
    v2 = _publish(tmp_path, [_chunk("c1", "d1"), _chunk("c2", "d2")], v1)
    assert (v1, v2) == ("v1", "v2")

    # the older version is still readable and still has its ORIGINAL contents
    _, old_chunks = index_store.load_version(v1, tmp_path)
    assert len(old_chunks) == 1
    _, new_chunks = index_store.load_version(v2, tmp_path)
    assert len(new_chunks) == 2


def test_manifest_records_what_the_version_contains(tmp_path) -> None:
    version = _publish(tmp_path, [_chunk("c1", "docA"), _chunk("c2", "docB")], None)
    manifest = index_store.read_manifest(version, tmp_path)
    assert manifest.chunk_count == 2
    assert manifest.document_ids == ["docA", "docB"]
    assert manifest.embedding_model


def test_index_and_chunks_always_agree(tmp_path) -> None:
    """The positional row<->line contract must hold for every published version."""
    version = _publish(tmp_path, [_chunk(f"c{i}", "d1") for i in range(5)], None)
    index, chunks = index_store.load_version(version, tmp_path)
    assert index.ntotal == len(chunks) == 5


def test_stale_base_version_is_rejected(tmp_path) -> None:
    """A writer that started from v1 must not overwrite a v2 published meanwhile."""
    v1 = _publish(tmp_path, [_chunk("c1", "d1")], None)
    _publish(tmp_path, [_chunk("c2", "d2")], v1)  # another writer reaches v2

    with pytest.raises(index_store.ConcurrentUpdateError):
        _publish(tmp_path, [_chunk("c3", "d3")], v1)  # still thinks head is v1


def test_concurrent_ingestion_keeps_every_document(tmp_path, monkeypatch) -> None:
    """The lost-update scenario, exercised for real with threads.

    Two uploads racing through the full merge path must both survive: the
    previous read-modify-write let the second writer overwrite the first.
    """
    from app.rag import ingest_runtime

    monkeypatch.setattr(
        ingest_runtime.embeddings,
        "embed_texts",
        lambda texts: np.random.rand(len(texts), 8).astype(np.float32),
    )
    _publish(tmp_path, [], None)  # start from an empty published index

    errors: list[Exception] = []

    def upload(doc_id: str) -> None:
        try:
            ingest_runtime._rebuild_and_publish(
                tmp_path, doc_id, [_chunk(f"{doc_id}-c1", doc_id)]
            )
        except Exception as exc:  # noqa: BLE001 - surfaced via `errors` below
            errors.append(exc)

    threads = [threading.Thread(target=upload, args=(f"doc{i}",)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent ingestion raised: {errors}"

    _, chunks, _ = index_store.load_current(tmp_path)
    surviving = {c.doc_id for c in chunks}
    assert surviving == {f"doc{i}" for i in range(6)}, "a concurrent upload was lost"


def test_prune_keeps_current_and_recent(tmp_path) -> None:
    version = None
    for i in range(6):
        version = _publish(tmp_path, [_chunk(f"c{i}", "d1")], version)
    removed = index_store.prune_old_versions(keep=2, root=tmp_path)

    assert index_store.read_pointer(tmp_path) == version
    index_store.load_version(version, tmp_path)  # current still loads
    assert "v1" in removed and version not in removed
