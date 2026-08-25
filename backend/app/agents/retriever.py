"""Retriever node: hybrid dense+sparse search fused with RRF.

Role in architecture: loads the FAISS/BM25 artifacts built in P1 once per
process (module-level globals — reused across warm Lambda invocations) and
turns a query into ranked Chunk objects. Runs again after repair_rewrite
with the rewritten query.
"""

import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.agents.budget import check_budget
from app.agents.state import AgentState
from app.config import get_settings
from app.models.schemas import Chunk
from app.rag import bm25_store, embeddings, index_store

_index: Any = None
_chunks: list[Chunk] = []
_bm25: Any = None
_version: str | None = None

_TERM_RE = re.compile(r"\w+")


def _normalize_query_terms(query: str) -> set[str]:
    return {term for term in _TERM_RE.findall(query.lower()) if len(term) > 1}


def _term_matches(term: str, text: str) -> bool:
    tokens = _TERM_RE.findall(text.lower())
    return any(term in token or token in term for token in tokens)


def _rerank_candidates(
    query: str,
    chunks: list[Chunk],
    row_ids: list[int],
    query_terms: set[str] | None = None,
) -> list[Chunk]:
    """Boost chunks whose section path or text aligns tightly with the query."""
    if not chunks:
        return []

    terms = query_terms or _normalize_query_terms(query)
    scored: list[tuple[float, int, Chunk]] = []
    for rank, chunk in enumerate(chunks):
        row_id = row_ids[rank] if rank < len(row_ids) else rank
        combined_text = f"{chunk.section_path}\n\n{chunk.text}".lower()
        section_text = chunk.section_path.lower()

        overlap = sum(1 for term in terms if _term_matches(term, combined_text))
        section_overlap = sum(1 for term in terms if _term_matches(term, section_text))
        exact_phrase_bonus = 1.0 if query.lower() in combined_text else 0.0
        section_bonus = 0.8 if section_overlap else 0.0
        positional_bonus = max(0.0, 1.0 - (rank / max(1, len(chunks))))

        score = (
            overlap
            + (section_overlap * 1.5)
            + section_bonus
            + exact_phrase_bonus
            + positional_bonus
        )
        scored.append((score, -row_id, chunk))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunk for _, _, chunk in scored]


def reset_cache() -> None:
    """Drop the in-memory index so the next query reloads fresh from S3.

    Called after an upload merges a new document — otherwise a warm Lambda
    would keep serving the pre-upload index until its next cold start.
    """
    global _index, _chunks, _bm25, _version
    _index, _chunks, _bm25, _version = None, [], None, None


def _ensure_loaded() -> None:
    global _index, _chunks, _bm25, _version
    if _index is None:
        settings = get_settings()
        root = Path(settings.index_dir)
        if settings.use_s3_index:
            # Lambda: artifacts live in S3; /tmp is the only writable path.
            # Downloaded once per cold start, reused by every warm invocation.
            root = Path("/tmp/index")
            index_store.load_current_from_s3(root)
        loaded = index_store.load_current(root)
        if loaded is None:  # nothing published yet — an empty corpus, not an error
            _index, _chunks, _bm25, _version = index_store.build_faiss(
                np.zeros((0, 384), dtype=np.float32)
            ), [], None, None
            return
        _index, _chunks, version = loaded
        _bm25 = index_store.load_bm25(version, root)
        _version = version


def _visible_rows(owner_id: str, doc_id: str | None) -> set[int] | None:
    """Row ids this caller is allowed to see.

    Returns None when no filtering is needed (no rows are excluded), so the
    common unscoped path stays allocation-free.

    Two independent restrictions apply:
      * tenant isolation — a chunk is only visible to the user who owns the
        document it came from. This is enforced here, at retrieval, rather
        than only at the API edge, so no future caller can bypass it.
      * document scope — when the client asks about one document, matching is
        on EXACT document_id. The previous `startswith` test meant one
        document id that happened to prefix another silently widened the
        scope to both.
    """
    if doc_id is None and not owner_id:
        return None
    rows = set()
    for i, chunk in enumerate(_chunks):
        # Chunks written before owner tracking have owner_id="" and are treated
        # as shared corpus (the seeded demo documents), visible to everyone.
        if chunk.owner_id and owner_id and chunk.owner_id != owner_id:
            continue
        if doc_id is not None and chunk.doc_id != doc_id:
            continue
        rows.add(i)
    return rows


def retriever_node(state: AgentState) -> AgentState:
    if not check_budget(state):
        state["status"] = "refused"
        return state

    settings = get_settings()
    t0 = time.perf_counter()
    _ensure_loaded()

    doc_id = state.get("doc_id")
    owner_id = state.get("user_id", "")
    allowed = _visible_rows(owner_id, doc_id)

    # When scoping/filtering, pull the whole candidate space before filtering so
    # a small document is fully covered rather than being crowded out of a
    # globally-ranked top-N by a larger corpus.
    cand = _index.ntotal if allowed is not None else settings.candidates_per_retriever

    if _index.ntotal == 0 or (allowed is not None and not allowed):
        state["retrieved"] = []
        state["trace"].record_step(
            "retriever", int((time.perf_counter() - t0) * 1000), chunks=0, scope=doc_id or "all"
        )
        return state

    qvec = embeddings.embed_texts([state["query"]])[0]
    dense = [row for row, _ in index_store.search(_index, qvec, cand)]
    sparse = [row for row, _ in bm25_store.search(_bm25, state["query"], cand)]
    if allowed is not None:
        dense = [r for r in dense if r in allowed]
        sparse = [r for r in sparse if r in allowed]

    fused = bm25_store.rrf_fuse([dense, sparse], k=settings.rrf_k, top_k=settings.top_k)
    ranked_rows = [row for row, _ in fused]
    reranked = _rerank_candidates(
        state["query"],
        [_chunks[row] for row in ranked_rows],
        ranked_rows,
        query_terms=_normalize_query_terms(state["query"]),
    )

    state["retrieved"] = reranked[: settings.top_k]

    duration_ms = int((time.perf_counter() - t0) * 1000)
    state["trace"].record_step(
        "retriever",
        duration_ms,
        chunks=len(state["retrieved"]),
        scope=doc_id or "all",
        index_version=_version or "none",
    )
    return state
