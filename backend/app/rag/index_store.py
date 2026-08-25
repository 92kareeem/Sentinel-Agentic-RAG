"""Versioned, atomically-published index storage.

Role in architecture: the previous layout wrote faiss.index, chunks.jsonl and
bm25.pkl in place, one file at a time, into a single directory. Two problems
followed from that, both of which lose user data:

  1. Torn publication. A reader (or an S3 sync) could observe a NEW
     faiss.index next to an OLD chunks.jsonl. Row i of the index then names
     the wrong chunk, so retrieval returns confidently mismatched text — and
     because the row/line contract is positional, nothing detects it.

  2. Lost updates. Ingestion was read-modify-write with no concurrency
     control. Two uploads that both read version N each write their own N+1,
     and whichever lands second silently erases the other user's document.

This module fixes both by making every publication a new immutable version
directory plus an atomic pointer swap:

    index/v3/{faiss.index,chunks.jsonl,bm25.pkl,manifest.json}
    index/current.json      -> {"version": "v3"}

A version directory is only ever pointed at once it is completely written, so
there is no torn state to observe. Concurrent writers are serialized by a
compare-and-set on the pointer: a writer that based its work on a version that
is no longer current loses the race and retries against the new head, rather
than overwriting it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings
from app.models.schemas import Chunk

INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.jsonl"
BM25_FILE = "bm25.pkl"
MANIFEST_FILE = "manifest.json"
POINTER_FILE = "current.json"

ARTIFACTS = (INDEX_FILE, CHUNKS_FILE, BM25_FILE, MANIFEST_FILE)


class ConcurrentUpdateError(RuntimeError):
    """Another writer published while this one was working; retry on the new head."""


@dataclass(frozen=True)
class IndexVersion:
    version: str
    chunk_count: int
    document_ids: list[str]
    created_at: str
    embedding_model: str

    @property
    def number(self) -> int:
        return int(self.version.lstrip("v"))


def build_faiss(vectors: np.ndarray) -> Any:
    """Brute-force inner-product index; exact, right-sized for <1M vectors.

    Vectors are L2-normalized upstream, so inner product == cosine similarity.
    """
    import faiss  # lazy: heavy native import

    # shape[1], not .size: a zero-row array still declares its width, and
    # `.size` is 0 for (0, 8) — reading width from `.size` would silently
    # rebuild an empty index at the wrong dimension.
    dim = vectors.shape[1] if vectors.ndim == 2 and vectors.shape[1] else 384
    index = faiss.IndexFlatIP(dim)
    if vectors.shape[0]:
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return index


def search(index: Any, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return [(row, score)] best-first. Row maps into the chunks list."""
    if index.ntotal == 0:
        return []
    scores, rows = index.search(query_vec.reshape(1, -1), min(k, index.ntotal))
    return [(int(r), float(s)) for r, s in zip(rows[0], scores[0], strict=True) if r >= 0]


def _root() -> Path:
    return Path(get_settings().index_dir)


def _version_dir(root: Path, version: str) -> Path:
    return root / version


def read_pointer(root: Path | None = None) -> str | None:
    """Current published version, or None if the index has never been built."""
    root = root or _root()
    pointer = root / POINTER_FILE
    if not pointer.exists():
        return None
    try:
        version: str = json.loads(pointer.read_text(encoding="utf-8"))["version"]
        return version
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def _write_pointer(root: Path, version: str, *, expected: str | None) -> None:
    """Compare-and-set the current-version pointer.

    The check-then-write below is not atomic against another OS process, which
    is why AWS mode additionally serializes publication through a DynamoDB
    conditional update (see documents/ingest_lock.py). Within one process it
    is guarded by the caller's lock. The re-read here still catches the common
    case cheaply and keeps local development honest.
    """
    root.mkdir(parents=True, exist_ok=True)
    actual = read_pointer(root)
    if actual != expected:
        raise ConcurrentUpdateError(
            f"index moved from {expected} to {actual} while publishing {version}"
        )
    pointer = root / POINTER_FILE
    tmp = pointer.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"version": version}), encoding="utf-8")
    os.replace(tmp, pointer)  # atomic on POSIX and Windows


def load_version(version: str, root: Path | None = None) -> tuple[Any, list[Chunk]]:
    """Load one specific immutable index version."""
    import faiss

    root = root or _root()
    vdir = _version_dir(root, version)
    index = faiss.read_index(str(vdir / INDEX_FILE))
    with (vdir / CHUNKS_FILE).open(encoding="utf-8") as f:
        chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
    if index.ntotal != len(chunks):
        # Defense in depth: atomic publication should make this unreachable.
        raise RuntimeError(
            f"index/chunks misaligned in {version}: "
            f"{index.ntotal} vectors vs {len(chunks)} chunks"
        )
    return index, chunks


def load_current(root: Path | None = None) -> tuple[Any, list[Chunk], str] | None:
    """Load the currently published version, or None if nothing is published."""
    root = root or _root()
    version = read_pointer(root)
    if version is None:
        return None
    index, chunks = load_version(version, root)
    return index, chunks, version


def load_bm25(version: str, root: Path | None = None) -> Any:
    import pickle

    root = root or _root()
    with (_version_dir(root, version) / BM25_FILE).open("rb") as f:
        return pickle.load(f)  # noqa: S301 - artifact is built by us, not user input


def read_manifest(version: str, root: Path | None = None) -> IndexVersion:
    root = root or _root()
    raw = json.loads((_version_dir(root, version) / MANIFEST_FILE).read_text(encoding="utf-8"))
    return IndexVersion(**raw)


def next_version(current: str | None) -> str:
    return "v1" if current is None else f"v{int(current.lstrip('v')) + 1}"


def publish(
    index: Any,
    chunks: list[Chunk],
    bm25: Any,
    *,
    base_version: str | None,
    root: Path | None = None,
) -> str:
    """Write a complete new version, then atomically point at it.

    base_version is the version this build was derived from; if the pointer has
    moved since, publication is refused so the concurrent writer's work is not
    silently discarded.
    """
    import pickle

    import faiss

    root = root or _root()
    version = next_version(read_pointer(root))
    vdir = _version_dir(root, version)
    vdir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(vdir / INDEX_FILE))
    with (vdir / CHUNKS_FILE).open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")
    with (vdir / BM25_FILE).open("wb") as f:
        pickle.dump(bm25, f)

    manifest = IndexVersion(
        version=version,
        chunk_count=len(chunks),
        document_ids=sorted({c.doc_id for c in chunks}),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        embedding_model=get_settings().embed_model_name,
    )
    (vdir / MANIFEST_FILE).write_text(json.dumps(manifest.__dict__, indent=2), encoding="utf-8")

    # Only now, with every artifact durable, does the version become visible.
    _write_pointer(root, version, expected=base_version)
    return version


def prune_old_versions(keep: int = 3, root: Path | None = None) -> list[str]:
    """Delete all but the newest `keep` versions.

    Old versions are retained (not deleted on publish) so an in-flight reader
    holding version N keeps working while N+1 is published, and so a bad
    publish can be rolled back by repointing. Returns the versions removed.
    """
    import shutil

    root = root or _root()
    if not root.exists():
        return []
    current = read_pointer(root)
    versions = sorted(
        (p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("v")),
        key=lambda v: int(v.lstrip("v")),
        reverse=True,
    )
    doomed = [v for v in versions[keep:] if v != current]
    for version in doomed:
        shutil.rmtree(_version_dir(root, version), ignore_errors=True)
    return doomed


# ---------------------------------------------------------------- S3 sync


def sync_version_to_s3(version: str, root: Path | None = None) -> None:
    """Upload one immutable version, then move the remote pointer last.

    Ordering matters for the same reason as locally: the pointer is only
    updated once every artifact of that version is durable in S3, so a Lambda
    cold-starting mid-sync reads a complete older version rather than a torn
    newer one.
    """
    import boto3

    settings = get_settings()
    root = root or _root()
    s3 = boto3.client("s3", region_name=settings.aws_region)
    vdir = _version_dir(root, version)

    for name in ARTIFACTS:
        s3.upload_file(str(vdir / name), settings.s3_bucket_docs, f"index/{version}/{name}")
    s3.put_object(
        Bucket=settings.s3_bucket_docs,
        Key=f"index/{POINTER_FILE}",
        Body=json.dumps({"version": version}).encode(),
        ContentType="application/json",
    )


def load_current_from_s3(dest_root: Path) -> str:
    """Download the currently published version (Lambda: dest under /tmp)."""
    import boto3

    settings = get_settings()
    s3 = boto3.client("s3", region_name=settings.aws_region)

    pointer = json.loads(
        s3.get_object(Bucket=settings.s3_bucket_docs, Key=f"index/{POINTER_FILE}")["Body"]
        .read()
        .decode()
    )
    version: str = pointer["version"]

    vdir = _version_dir(dest_root, version)
    vdir.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        s3.download_file(
            settings.s3_bucket_docs, f"index/{version}/{name}", str(vdir / name)
        )
    (dest_root / POINTER_FILE).write_text(json.dumps({"version": version}), encoding="utf-8")
    return version
