"""Retriever node: hybrid dense+sparse search fused with RRF.

Role in architecture: turns a query into ranked Chunk objects, scoped to what
the caller is allowed to see. The index itself is owned by
rag/retriever_snapshot.py, which hands out one immutable, self-consistent view
per request; this module binds exactly one of those on entry and never reads
a global mid-search. Runs again after repair_rewrite with the rewritten
search query.

Authorization happens BEFORE scoring. The allowed rows are handed to FAISS as
a selector and to BM25 as a mask, so a chunk the caller may not see is never
ranked, never loaded into state and never in a prompt — not filtered out of a
top-N it had already won a place in. A final re-check on the returned chunks
is kept as a tripwire: it should never fire, and if it ever does the trace
says so.
"""

from __future__ import annotations

import re
import time

from app.agents.budget import check_budget
from app.agents.state import AgentState
from app.config import get_settings
from app.guardrails import access
from app.models.schemas import AccessContext, Chunk
from app.observability.logging import get_logger
from app.rag import bm25_store, embeddings, index_store, retriever_snapshot

_TERM_RE = re.compile(r"\w+")
_logger = get_logger()


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
    """Drop the cached index so the next query reloads it.

    Called after an upload merges a new document — otherwise a warm process
    would keep serving the pre-upload index until its next cold start. Kept as
    a thin alias so callers outside rag/ do not need to know where the
    snapshot lives.
    """
    retriever_snapshot.invalidate()


def _context(state: AgentState) -> AccessContext:
    """The caller's access context, or an owner-only one for legacy callers
    (the eval harness and scripts build state with just a user_id)."""
    ctx = state.get("access")
    if ctx is not None:
        return ctx
    return AccessContext(user_id=state.get("user_id", ""))


def retriever_node(state: AgentState) -> AgentState:
    if not check_budget(state):
        state["status"] = "refused"
        return state

    settings = get_settings()
    t0 = time.perf_counter()

    # Bound ONCE, then used for everything below. The index, the chunk list its
    # row ids point into, the BM25 model and the version only mean anything
    # together; they used to be four globals that an upload could swap out from
    # under a query already running in FastAPI's threadpool.
    snap = retriever_snapshot.current()

    doc_id = state.get("doc_id")
    ctx = _context(state)
    allowed = access.visible_rows(snap.chunks, ctx, doc_id)
    excluded = 0 if allowed is None else snap.size - len(allowed)

    # When filtering, pull the whole candidate space so a small document is
    # fully covered rather than crowded out of a globally-ranked top-N.
    cand = snap.index.ntotal if allowed is not None else settings.candidates_per_retriever

    if snap.index.ntotal == 0 or (allowed is not None and not allowed):
        state["retrieved"] = []
        state["trace"].record_step(
            "retriever",
            int((time.perf_counter() - t0) * 1000),
            chunks=0,
            scope=doc_id or "all",
            rows_excluded=excluded,
        )
        return state

    # search_query, not query: after repair_rewrite these differ, and it is
    # the reformulation that is keyword-dense enough to retrieve well. The
    # user's original wording stays in `query`, which is what the synthesizer
    # and the critic answer and score against.
    search_query = state["search_query"]
    qvec = embeddings.embed_texts([search_query])[0]
    dense = [row for row, _ in index_store.search(snap.index, qvec, cand, allowed_rows=allowed)]
    sparse = [
        row for row, _ in bm25_store.search(snap.bm25, search_query, cand, allowed_rows=allowed)
    ]

    fused = bm25_store.rrf_fuse([dense, sparse], k=settings.rrf_k, top_k=settings.top_k)
    ranked_rows = [row for row, _ in fused]
    reranked = _rerank_candidates(
        search_query,
        [snap.chunks[row] for row in ranked_rows],
        ranked_rows,
        query_terms=_normalize_query_terms(search_query),
    )

    # Tripwire. Pre-scoring exclusion makes this a no-op; it exists so that a
    # future regression in the selector path is loud instead of a leak.
    safe = [
        c for c in reranked
        if access.can_see(ctx, c) and (doc_id is None or c.doc_id == doc_id)
    ]
    if len(safe) != len(reranked):
        _logger.error(
            "access_tripwire",
            extra={"data": f"{len(reranked) - len(safe)} chunk(s) failed the final check"},
        )
    state["retrieved"] = safe[: settings.top_k]

    duration_ms = int((time.perf_counter() - t0) * 1000)
    state["trace"].record_step(
        "retriever",
        duration_ms,
        chunks=len(state["retrieved"]),
        scope=doc_id or "all",
        rows_excluded=excluded,
        index_version=snap.version or "none",
    )
    return state
