"""Dense vector store: FAISS IndexFlatIP + chunks.jsonl, synced with S3.

Role in architecture: exact cosine search (inner product over normalized
vectors) with zero servers. Index row i corresponds to line i of chunks.jsonl
— that positional alignment is the contract every reader relies on.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings
from app.models.schemas import Chunk

INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.jsonl"
BM25_FILE = "bm25.pkl"


def build_index(vectors: np.ndarray) -> Any:
    """Brute-force inner-product index; exact, right-sized for <1M vectors."""
    import faiss  # lazy: heavy native import

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save(index: Any, chunks: list[Chunk], index_dir: Path) -> None:
    """Persist index + chunks.jsonl. Row i of the index == line i of the jsonl."""
    import faiss

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / INDEX_FILE))
    with (index_dir / CHUNKS_FILE).open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")


def load(index_dir: Path) -> tuple[Any, list[Chunk]]:
    import faiss

    index = faiss.read_index(str(index_dir / INDEX_FILE))
    with (index_dir / CHUNKS_FILE).open(encoding="utf-8") as f:
        chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
    if index.ntotal != len(chunks):
        raise RuntimeError(
            f"index/chunks misaligned: {index.ntotal} vectors vs {len(chunks)} chunks"
        )
    return index, chunks


def search(index: Any, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return [(row, score)] best-first. Row maps into the chunks list."""
    scores, rows = index.search(query_vec.reshape(1, -1), k)
    return [(int(r), float(s)) for r, s in zip(rows[0], scores[0], strict=True) if r >= 0]


# ---------------------------------------------------------------- S3 sync (P4)


def sync_to_s3(index_dir: Path) -> None:
    """Upload index artifacts to s3://<docs-bucket>/index/. No-op config locally."""
    s = get_settings()
    import boto3

    s3 = boto3.client("s3", region_name=s.aws_region)
    for name in (INDEX_FILE, CHUNKS_FILE, BM25_FILE):
        s3.upload_file(str(index_dir / name), s.s3_bucket_docs, f"index/{name}")


def load_from_s3(dest_dir: Path) -> Path:
    """Cold-start download of index artifacts (Lambda: dest under /tmp)."""
    s = get_settings()
    import boto3

    s3 = boto3.client("s3", region_name=s.aws_region)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in (INDEX_FILE, CHUNKS_FILE, BM25_FILE):
        s3.download_file(s.s3_bucket_docs, f"index/{name}", str(dest_dir / name))
    return dest_dir
