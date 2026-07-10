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

    qvec = embeddings.embed_texts([state["query"]])[0]
    dense = [row for row, _ in faiss_store.search(_index, qvec, settings.candidates_per_retriever)]
    sparse = [
        row for row, _ in bm25_store.search(_bm25, state["query"], settings.candidates_per_retriever)
    ]
    fused = bm25_store.rrf_fuse([dense, sparse], k=settings.rrf_k, top_k=settings.top_k)

    state["retrieved"] = [_chunks[row] for row, _ in fused]

    duration_ms = int((time.perf_counter() - t0) * 1000)
    state["trace"].record_step("retriever", duration_ms, chunks=len(state["retrieved"]))
    return state
