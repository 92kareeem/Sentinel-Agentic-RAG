"""Retriever node: hybrid dense+sparse search fused with RRF.

Role in architecture: turns a query into ranked Chunk objects, scoped to what
the caller is allowed to see. The index itself is owned by
rag/retriever_snapshot.py, which hands out one immutable, self-consistent view
per request; this module binds exactly one of those on entry and never reads
a global mid-search. Runs again after repair_rewrite with the rewritten
search query.
"""

import re
import time

from app.agents.budget import check_budget
from app.agents.state import AgentState
from app.config import get_settings
from app.models.schemas import Chunk
from app.rag import bm25_store, embeddings, index_store, retriever_snapshot
from app.rag.retriever_snapshot import IndexSnapshot

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
    """Drop the cached index so the next query reloads it.

    Called after an upload merges a new document — otherwise a warm process
    would keep serving the pre-upload index until its next cold start. Kept as
    a thin alias so callers outside rag/ do not need to know where the
    snapshot lives.
    """
    retriever_snapshot.invalidate()


def _visible_rows(snap: IndexSnapshot, owner_id: str, doc_id: str | None) -> set[int] | None:
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
    for i, chunk in enumerate(snap.chunks):
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

    # Bound ONCE, then used for everything below. The index, the chunk list its
    # row ids point into, and the BM25 model only mean anything together; they
    # used to be three globals that an upload could swap out from under a
    # query already running in FastAPI's threadpool. The loud version of that
    # was an IndexError on `_chunks[row]`; the quiet version returned real
    # chunk text for row ids belonging to a different index version, so a
    # citation pointed at the wrong passage with nothing reporting a problem.
    snap = retriever_snapshot.current()

    doc_id = state.get("doc_id")
    owner_id = state.get("user_id", "")
    allowed = _visible_rows(snap, owner_id, doc_id)

    # When scoping/filtering, pull the whole candidate space before filtering so
    # a small document is fully covered rather than being crowded out of a
    # globally-ranked top-N by a larger corpus.
    cand = snap.index.ntotal if allowed is not None else settings.candidates_per_retriever

    if snap.index.ntotal == 0 or (allowed is not None and not allowed):
        state["retrieved"] = []
        state["trace"].record_step(
            "retriever", int((time.perf_counter() - t0) * 1000), chunks=0, scope=doc_id or "all"
        )
        return state

    # search_query, not query: after repair_rewrite these differ, and it is
    # the reformulation that is keyword-dense enough to retrieve well. The
    # user's original wording stays in `query`, which is what the synthesizer
    # and the critic answer and score against.
    search_query = state["search_query"]
    qvec = embeddings.embed_texts([search_query])[0]
    dense = [row for row, _ in index_store.search(snap.index, qvec, cand)]
    sparse = [row for row, _ in bm25_store.search(snap.bm25, search_query, cand)]
    if allowed is not None:
        dense = [r for r in dense if r in allowed]
        sparse = [r for r in sparse if r in allowed]

    fused = bm25_store.rrf_fuse([dense, sparse], k=settings.rrf_k, top_k=settings.top_k)
    ranked_rows = [row for row, _ in fused]
    reranked = _rerank_candidates(
        search_query,
        [snap.chunks[row] for row in ranked_rows],
        ranked_rows,
        query_terms=_normalize_query_terms(search_query),
    )

    state["retrieved"] = reranked[: settings.top_k]

    duration_ms = int((time.perf_counter() - t0) * 1000)
    state["trace"].record_step(
        "retriever",
        duration_ms,
        chunks=len(state["retrieved"]),
        scope=doc_id or "all",
        index_version=snap.version or "none",
    )
    return state
