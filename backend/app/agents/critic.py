"""Critic node: LLM-as-judge scoring of faithfulness and relevance.

Role in architecture: the graph's own quality gate, always run with the
cheap model regardless of which model wrote the answer — a judge only needs
to be consistent, not the strongest model, and using the same grader every
time keeps scores comparable across repair attempts.

The verdict is validated as a whole, not just parsed. "The model returned
valid JSON that is not a verdict" — an array, a score of "high", a score of
7 on a 0-1 scale — used to escape the parse-failure handling, raise deep in
the node, and surface as a 503 "upstream LLM error" after an answer had
already been drafted. Any malformed verdict now costs one re-ask and then
fails closed, and the tokens it spent are always accounted for.
"""

from __future__ import annotations

import json
import time

from pydantic import BaseModel, Field, ValidationError

from app.agents.budget import check_budget, llm_call
from app.agents.state import AgentState
from app.config import get_settings
from app.models.schemas import CriticScores

_SYSTEM_PROMPT = (
    "You are a strict grading judge. Given a question, retrieved context, and an "
    'answer, output ONLY JSON: {"faithfulness": <0-1>, "relevance": <0-1>, '
    '"why": "<one sentence>"}. faithfulness = is every claim supported by the '
    "context? relevance = does the answer address the question? An answer of "
    "INSUFFICIENT_CONTEXT that is truly warranted scores faithfulness=1.0."
)


class Verdict(BaseModel):
    """The only shape a judge reply is allowed to have."""

    faithfulness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    why: str = ""


class InvalidVerdict(ValueError):
    """The judge replied, but not with a verdict."""


def parse_verdict(content: str) -> Verdict:
    """Strict: a dict with two numbers in [0, 1], nothing else accepted.

    Pydantic coerces "0.9" to 0.9, which is fine, but refuses "high", a
    list, or 7 — the three malformed shapes seen from a real judge.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise InvalidVerdict(f"not JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise InvalidVerdict(f"expected an object, got {type(data).__name__}")
    try:
        return Verdict.model_validate(data)
    except ValidationError as exc:
        raise InvalidVerdict(str(exc.errors()[0].get("msg", "invalid"))) from exc


def critic_node(state: AgentState) -> AgentState:
    if not check_budget(state):
        state["status"] = "refused"
        return state

    settings = get_settings()
    t0 = time.perf_counter()
    context = "\n\n".join(c.text for c in state["retrieved"])
    tokens_in = tokens_out = 0

    verdict: Verdict | None = None
    last_problem = ""
    for _ in range(2):  # one re-ask on a malformed verdict
        out = llm_call(
            state,
            model=settings.model_simple,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question: {state['query']}\n\nContext:\n{context}\n\n"
                        f"Answer:\n{state['answer']}"
                    ),
                },
            ],
            max_tokens=300,
            json_mode=True,
        )
        if out is None:  # deadline ran out mid-call; llm_call already refused
            return state
        content, tin, tout = out
        # Accounted BEFORE parsing: a malformed reply still cost real tokens,
        # and the budget must see them or the request can overspend by a full
        # critic call per bad verdict.
        tokens_in += tin
        tokens_out += tout
        try:
            verdict = parse_verdict(content)
            break
        except InvalidVerdict as exc:
            last_problem = str(exc)
            continue

    if verdict is None:  # judge failed twice: fail closed, never silently pass
        verdict = Verdict(faithfulness=0.0, relevance=0.0, why=f"invalid verdict: {last_problem}")

    scores = CriticScores(faithfulness=verdict.faithfulness, relevance=verdict.relevance)
    state["critic"] = scores
    state["trace"].critic_scores.append({**scores.model_dump(), "attempt": state["attempt"]})
    state["token_budget_left"] -= tokens_in + tokens_out

    duration_ms = int((time.perf_counter() - t0) * 1000)
    state["trace"].record_step("critic", duration_ms, tokens_in, tokens_out, why=verdict.why)
    return state
