"""Synthesizer node: generates a cited answer from retrieved chunks only.

Role in architecture: the only node that produces the user-facing answer.
Prompt rules are load-bearing: answer ONLY from the given chunks, every
sentence ends with [chunk:<id>], and INSUFFICIENT_CONTEXT is a valid, honest
answer the critic must not penalize as a failure.
"""

import time

from app.agents.budget import check_budget
from app.agents.state import AgentState
from app.llm import groq_client
from app.models.schemas import Citation

_SYSTEM_PROMPT = (
    "You are Sentinel, a document Q&A assistant. Answer ONLY using the numbered "
    "context chunks below. Every sentence you write must end with a citation tag "
    "like [chunk:<chunk_id>] naming the chunk it came from. If the chunks do not "
    "contain enough information to answer, respond with exactly: INSUFFICIENT_CONTEXT"
)


def _build_context(chunks: list) -> str:
    return "\n\n".join(f"[chunk:{c.chunk_id}] ({c.section_path})\n{c.text}" for c in chunks)


def _extract_citations(answer: str, chunks: list) -> list[Citation]:
    by_id = {c.chunk_id: c for c in chunks}
    cited_ids = {cid for cid in by_id if f"[chunk:{cid}]" in answer}
    return [
        Citation(chunk_id=cid, section_path=by_id[cid].section_path, quote=by_id[cid].text[:200])
        for cid in cited_ids
    ]


def synthesizer_node(state: AgentState) -> AgentState:
    if not check_budget(state):
        state["status"] = "refused"
        return state

    t0 = time.perf_counter()
    context = _build_context(state["retrieved"])
    content, tokens_in, tokens_out = groq_client.chat_completion(
        model=state["model"],
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['query']}"},
        ],
        max_tokens=512,
    )
    state["answer"] = content.strip()
    state["citations"] = _extract_citations(state["answer"], state["retrieved"])
    state["token_budget_left"] -= tokens_in + tokens_out

    duration_ms = int((time.perf_counter() - t0) * 1000)
    state["trace"].record_step("synthesizer", duration_ms, tokens_in, tokens_out, model=state["model"])
    return state
