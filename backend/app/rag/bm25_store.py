"""Sparse keyword index (BM25) + reciprocal rank fusion.

Role in architecture: catches exact-term queries (ids, codes, names) that
embeddings blur. RRF lives here so the retriever agent node stays a thin
orchestration layer. Fusion is rank-based on purpose: cosine and BM25 scores
share no scale, ranks do.
"""

import pickle
import re
from pathlib import Path
from typing import Any

BM25_FILE = "bm25.pkl"

_WORD_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens; must be identical at build and query time."""
    return _WORD_RE.findall(text.lower())


def build(corpus_texts: list[str]) -> Any:
    from rank_bm25 import BM25Okapi  # lazy import, symmetry with faiss

    return BM25Okapi([tokenize(t) for t in corpus_texts])


def save(bm25: Any, index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / BM25_FILE).open("wb") as f:
        pickle.dump(bm25, f)


def load(index_dir: Path) -> Any:
    with (index_dir / BM25_FILE).open("rb") as f:
        return pickle.load(f)  # noqa: S301 - artifact is built by us, not user input


def search(bm25: Any, query: str, k: int) -> list[tuple[int, float]]:
    """Return [(row, score)] best-first; rows align with build() corpus order."""
    scores = bm25.get_scores(tokenize(query))
    best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(i, float(scores[i])) for i in best]


def rrf_fuse(
    rankings: list[list[int]], k: int = 60, top_k: int = 8
) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(row) = sum over rankings of 1/(k + rank).

    Rank-based, so no cross-retriever score calibration is needed. Returns
    [(row, fused_score)] best-first, truncated to top_k.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking):
            fused[row] = fused.get(row, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return ordered[:top_k]
