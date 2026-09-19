"""The retrieval index as an immutable, atomically-swapped snapshot.

Role in architecture: retrieval reads four things that only mean anything
TOGETHER — the FAISS index, the chunk list its row ids point into, the BM25
model built over those same chunks, and the version they all came from. They
used to be four module globals, mutated independently by reset_cache() and
repopulated one assignment at a time by the loader.

That is a torn read waiting to happen. FastAPI runs `def` endpoints in a
threadpool, so an upload calling reset_cache() interleaves with a query
already inside retrieval: the query reads an index with 400 vectors, then
reads a chunk list that has just been emptied, and `_chunks[row]` raises
IndexError. The quieter version is worse — after a reload the row ids are
valid but belong to a DIFFERENT index version, so retrieval returns real,
confidently-cited chunk text for rows that mean something else. The citation
would point at the wrong passage with nothing anywhere reporting a problem.

The fix is to make the unit of consistency explicit. A snapshot is frozen and
built completely before it is published; a request binds one on entry and uses
only that one. Swapping the global cannot disturb a request in flight, because
nothing the request is reading is ever mutated — the old snapshot simply stays
alive until the last request holding it finishes.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings
from app.models.schemas import Chunk
from app.rag import index_store

_EMBED_DIM = 384


@dataclass(frozen=True)
class IndexSnapshot:
    """One self-consistent view of the corpus. Never mutated after construction."""

    index: Any
    chunks: tuple[Chunk, ...]
    bm25: Any
    version: str | None
    loaded_at: float

    @property
    def size(self) -> int:
        return len(self.chunks)


_snapshot: IndexSnapshot | None = None

# Guards construction, not reading. Same double-checked pattern as
# rag/embeddings.py: loading an index is seconds of work, and letting four
# concurrent cold requests each do it independently wastes memory the Lambda
# does not have.
_load_lock = threading.Lock()

# When the pointer was last consulted. Kept here rather than on the snapshot
# so the snapshot stays genuinely immutable — a "frozen" object that a
# staleness check writes back into is not a snapshot, and reasoning about it
# gets subtle in exactly the concurrent conditions this module exists for.
_last_pointer_check: float = 0.0


def _empty() -> IndexSnapshot:
    """A published-nothing corpus. Not an error — a new account has no documents."""
    return IndexSnapshot(
        index=index_store.build_faiss(np.zeros((0, _EMBED_DIM), dtype=np.float32)),
        chunks=(),
        bm25=None,
        version=None,
        loaded_at=time.monotonic(),
    )


def _build() -> IndexSnapshot:
    settings = get_settings()
    root = Path(settings.index_dir)
    if settings.use_s3_index:
        # Lambda: artifacts live in S3 and /tmp is the only writable path.
        root = Path("/tmp/index")
        index_store.load_current_from_s3(root)
    loaded = index_store.load_current(root)
    if loaded is None:
        return _empty()
    index, chunks, version = loaded
    return IndexSnapshot(
        index=index,
        chunks=tuple(chunks),
        bm25=index_store.load_bm25(version, root),
        version=version,
        loaded_at=time.monotonic(),
    )


def _is_stale(snap: IndexSnapshot) -> bool:
    """Has someone else published since this snapshot was built?

    A warm process never learns about an upload handled by a DIFFERENT process
    — reset_cache() only clears the cache of the instance that served the
    upload. Without this, a user uploads a document, the next query lands on
    another warm instance, and the document appears to be missing; they upload
    it again, and now it is indexed twice.

    Checked at most once per index_refresh_seconds, and only when a query
    actually arrives. That keeps the extra work to one small pointer read per
    refresh window on an otherwise idle system — deliberately not a background
    poller, which would bill for staying awake with nobody asking anything.
    """
    global _last_pointer_check
    ttl = get_settings().index_refresh_seconds
    if ttl <= 0:  # opt out entirely (tests, and single-process local runs)
        return False
    now = time.monotonic()
    if now - _last_pointer_check < ttl:
        return False
    _last_pointer_check = now  # set before the read, so a failing read still backs off
    published = index_store.read_published_version()
    if published is None:
        # Nothing published, or the check itself failed. Neither is a reason to
        # throw away a working index.
        return False
    return published != snap.version


def current() -> IndexSnapshot:
    """The snapshot a request should bind for its whole lifetime.

    Callers must hold the returned object rather than calling this repeatedly:
    two calls can legitimately return different snapshots, and mixing row ids
    across them is exactly the inconsistency this module exists to prevent.
    """
    global _snapshot, _last_pointer_check
    snap = _snapshot
    if snap is not None and not _is_stale(snap):
        return snap
    with _load_lock:
        # Re-check: another thread may have loaded while this one waited.
        if _snapshot is None or _snapshot is snap:
            _snapshot = _build()
            _last_pointer_check = time.monotonic()
        return _snapshot


def invalidate() -> None:
    """Drop the cached snapshot so the next query rebuilds from the pointer.

    Called after this process publishes a new version. Requests already
    holding the old snapshot are unaffected and finish against consistent
    data — which is the point of the swap being a single assignment.
    """
    global _snapshot, _last_pointer_check
    _snapshot = None
    _last_pointer_check = 0.0
