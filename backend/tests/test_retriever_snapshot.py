"""The index snapshot: one self-consistent view per request.

The index, the chunk list its row ids point into, and the BM25 model built
over those chunks only mean anything together. They used to be separate
module globals that an upload could clear and repopulate one assignment at a
time, while a query was already reading them from FastAPI's threadpool.

The loud symptom was IndexError on `_chunks[row]`. The quiet one is worse:
after a reload the row ids are valid but belong to a different index version,
so retrieval returns real chunk text for rows that mean something else and
the citation points at the wrong passage with nothing reporting a problem.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest
from app.config import get_settings
from app.models.schemas import Chunk
from app.rag import bm25_store, index_store, retriever_snapshot


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id="d",
        section_path="S",
        text=text,
        token_count=2,
        char_start=0,
        char_end=len(text),
    )


def _publish(root: Any, chunks: list[Chunk], base: str | None) -> str:
    vecs = np.ones((len(chunks), 384), dtype=np.float32)
    return index_store.publish(
        index_store.build_faiss(vecs),
        chunks,
        bm25_store.build([c.embed_text for c in chunks]),
        base_version=base,
        root=root,
    )


@pytest.fixture()
def index_root(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_MODE", "true")
    monkeypatch.setenv("USE_S3_INDEX", "false")
    get_settings.cache_clear()
    retriever_snapshot.invalidate()
    yield tmp_path
    retriever_snapshot.invalidate()
    get_settings.cache_clear()


def test_a_snapshot_is_internally_consistent(index_root: Any) -> None:
    _publish(index_root, [_chunk("a", "alpha text"), _chunk("b", "beta text")], None)

    snap = retriever_snapshot.current()

    assert snap.index.ntotal == len(snap.chunks) == 2
    assert snap.version is not None


def test_publishing_mid_request_cannot_change_the_snapshot_already_bound(
    index_root: Any,
) -> None:
    """THE regression test. A request binds one snapshot and finishes against
    it; the swap is a single assignment, so nothing it is reading is ever
    mutated underneath it."""
    v1 = _publish(index_root, [_chunk("a", "alpha")], None)
    held = retriever_snapshot.current()
    assert held.version == v1

    # An upload lands on this same process while the request is in flight.
    _publish(index_root, [_chunk("a", "alpha"), _chunk("b", "beta")], v1)
    retriever_snapshot.invalidate()

    # The in-flight request's view is untouched — every row id it holds still
    # resolves against the chunk list it was built with.
    assert len(held.chunks) == 1
    assert held.chunks[0].chunk_id == "a"
    assert held.index.ntotal == len(held.chunks)

    # ...and the next request sees the new version.
    assert len(retriever_snapshot.current().chunks) == 2


def test_concurrent_readers_never_see_a_half_loaded_index(index_root: Any) -> None:
    """Every thread must get an index whose row count matches its chunk list.
    With four independent globals, a reader could observe the new index and
    the old chunks."""
    _publish(index_root, [_chunk(str(i), f"text {i}") for i in range(20)], None)
    retriever_snapshot.invalidate()

    seen: list[tuple[int, int]] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(8)

    def reader() -> None:
        try:
            barrier.wait()
            snap = retriever_snapshot.current()
            seen.append((snap.index.ntotal, len(snap.chunks)))
        except Exception as exc:  # noqa: BLE001 — surfaced via `errors`
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert seen == [(20, 20)] * 8


def test_an_unpublished_corpus_is_an_empty_snapshot_not_an_error(
    index_root: Any,
) -> None:
    """A new account has no documents. That is a normal state, not a 500."""
    snap = retriever_snapshot.current()

    assert snap.index.ntotal == 0
    assert snap.chunks == ()
    assert snap.version is None


def test_a_version_published_by_another_process_is_picked_up(
    index_root: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A warm process never hears about an upload another instance handled —
    invalidate() only clears the cache of the instance that served it. Without
    a pointer re-check the user uploads a document, the next query lands
    elsewhere, it looks missing, and they upload it again."""
    monkeypatch.setenv("INDEX_REFRESH_SECONDS", "0")  # start with checks off
    get_settings.cache_clear()
    v1 = _publish(index_root, [_chunk("a", "alpha")], None)
    assert retriever_snapshot.current().version == v1

    # Another process publishes. This one is never told.
    _publish(index_root, [_chunk("a", "alpha"), _chunk("b", "beta")], v1)
    assert retriever_snapshot.current().version == v1, "should still be stale here"

    monkeypatch.setenv("INDEX_REFRESH_SECONDS", "1")
    get_settings.cache_clear()
    retriever_snapshot._last_pointer_check = 0.0  # as if the window had elapsed

    assert len(retriever_snapshot.current().chunks) == 2


def test_a_failed_pointer_check_keeps_serving_the_index_we_have(
    index_root: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Staleness is recoverable; a 503 on every query because the pointer read
    blipped is not."""
    monkeypatch.setenv("INDEX_REFRESH_SECONDS", "1")
    get_settings.cache_clear()
    _publish(index_root, [_chunk("a", "alpha")], None)
    before = retriever_snapshot.current()

    monkeypatch.setattr(index_store, "read_published_version", lambda: None)
    retriever_snapshot._last_pointer_check = 0.0

    assert retriever_snapshot.current() is before
