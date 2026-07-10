"""Critic node: LLM-as-judge scoring of faithfulness and relevance.

Role in architecture: the graph's own quality gate, always run with the
cheap model regardless of which model wrote the answer — a judge only needs
to be consistent, not the strongest model, and using the same grader every
time keeps scores comparable across repair attempts. Strict JSON parsing
with exactly one re-ask: "the model didn't return valid JSON" is a real,
common failure mode that must not crash the request.
"""

import json
import time

from app.agents.budget import check_budget
from app.agents.state import AgentState
from app.config import get_settings
from app.llm import groq_client
from app.models.schemas import CriticScores

_SYSTEM_PROMPT = (
    "You are a strict grading judge. Given a question, retrieved context, and an "
    'answer, output ONLY JSON: {"faithfulness": <0-1>, "relevance": <0-1>, '
    '"why": "<one sentence>"}. faithfulness = is every claim supported by the '
    "context? relevance = does the answer address the question? An answer of "
    "INSUFFICIENT_CONTEXT that is truly warranted scores faithfulness=1.0."
)


def _judge_once(query: str, context: str, answer: str, model: str) -> dict:
    content, tokens_in, tokens_out = groq_client.chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {query}\n\nContext:\n{context}\n\nAnswer:\n{answer}",
            },
        ],
        max_tokens=200,
        json_mode=True,
    )
    return {"data": json.loads(content), "tokens_in": tokens_in, "tokens_out": tokens_out}


def critic_node(state: AgentState) -> AgentState:
    if not check_budget(state):
        state["status"] = "refused"
        return state

    settings = get_settings()
    t0 = time.perf_counter()
    context = "\n\n".join(c.text for c in state["retrieved"])
    tokens_in = tokens_out = 0

    result = None
    for _ in range(2):  # one re-ask on parse failure
        try:
            out = _judge_once(state["query"], context, state["answer"], settings.groq_model_simple)
            tokens_in += out["tokens_in"]
            tokens_out += out["tokens_out"]
            result = out["data"]
            break
        except (json.JSONDecodeError, KeyError):
            continue

    if result is None:  # judge failed twice: fail closed, never silently pass
        result = {"faithfulness": 0.0, "relevance": 0.0, "why": "judge parse failure"}

    scores = CriticScores(
        faithfulness=float(result.get("faithfulness", 0.0)),
        relevance=float(result.get("relevance", 0.0)),
    )
    state["critic"] = scores
    state["trace"].critic_scores.append({**scores.model_dump(), "attempt": state["attempt"]})
    state["token_budget_left"] -= tokens_in + tokens_out

    duration_ms = int((time.perf_counter() - t0) * 1000)
    state["trace"].record_step("critic", duration_ms, tokens_in, tokens_out, why=result.get("why", ""))
    return state
