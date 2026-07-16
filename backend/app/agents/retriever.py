"""Retriever node: hybrid dense+sparse search fused with RRF.

Role in architecture: loads the FAISS/BM25 artifacts built in P1 once per
process (module-level globals — reused across warm Lambda invocations) and
turns a query into ranked Chunk objects. Runs again after repair_rewrite
with the rewritten query.
"""

import time
from typing import Any

from app.agents.budget import check_budget
from app.agents.state import AgentState
from app.config import get_settings
from app.models.schemas import Chunk
from app.rag import bm25_store, embeddings, faiss_store

_index: Any = None
_chunks: list[Chunk] = []
_bm25: Any = None


def reset_cache() -> None:
    """Drop the in-memory index so the next query reloads fresh from S3.

    Called after an upload merges a new document — otherwise a warm Lambda
    would keep serving the pre-upload index until its next cold start.
    """
    global _index, _chunks, _bm25
    _index, _chunks, _bm25 = None, [], None


def _ensure_loaded() -> None:
    global _index, _chunks, _bm25
    if _index is None:
        settings = get_settings()
        index_dir = settings.index_dir
        if settings.use_s3_index:
            # Lambda: artifacts live in S3; /tmp is the only writable path.
            # Downloaded once per cold start, reused by every warm invocation.
            from pathlib import Path

            index_dir = faiss_store.load_from_s3(Path("/tmp/index"))
        _index, _chunks = faiss_store.load(index_dir)
        _bm25 = bm25_store.load(index_dir)


def retriever_node(state: AgentState) -> AgentState:
    if not check_budget(state):
        state["status"] = "refused"
        return state

    settings = get_settings()
    t0 = time.perf_counter()
    _ensure_loaded()

    # Scope to one uploaded document when the caller asks. The index is shared
    # across all documents, so without this a vague question can match a
    # different document than the one the user just uploaded. When scoping, pull
    # the whole candidate space and filter, so a small document is fully covered.
    doc_id = state.get("doc_id")
    cand = _index.ntotal if doc_id else settings.candidates_per_retriever

    qvec = embeddings.embed_texts([state["query"]])[0]
    dense = [row for row, _ in faiss_store.search(_index, qvec, cand)]
    sparse = [row for row, _ in bm25_store.search(_bm25, state["query"], cand)]
    if doc_id:
        dense = [r for r in dense if _chunks[r].doc_id.startswith(doc_id)]
        sparse = [r for r in sparse if _chunks[r].doc_id.startswith(doc_id)]
    fused = bm25_store.rrf_fuse([dense, sparse], k=settings.rrf_k, top_k=settings.top_k)

    state["retrieved"] = [_chunks[row] for row, _ in fused]

    duration_ms = int((time.perf_counter() - t0) * 1000)
    state["trace"].record_step(
        "retriever", duration_ms, chunks=len(state["retrieved"]), scope=doc_id or "all"
    )
    return state
