"""Eval runner: golden dataset -> agent graph -> metrics -> evals/report.md.

Usage (from repo root, venv python):
    python evals/run.py            # full run against local index + live Groq

Exit code 1 if gates fail (mean faithfulness < 0.75 or unanswerable-refusal
< 2/3) so CI can use this directly.
"""

import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

# Set BEFORE app.config is imported: Settings is lru_cached, so a later change
# would not be seen.
#
# The harness is a batch job on a free tier whose token bucket refills over a
# minute. The interactive default (3 attempts) is tuned for a person waiting
# and is not enough time for that bucket to refill, so a suite that is merely
# SLOW reads as a suite that is BROKEN — measured: 14 of 17 answerable
# questions "refused" in under 120 ms each. Waiting is the correct response to
# a rate limit when nobody is waiting on you.
os.environ.setdefault("LLM_MAX_RETRIES", "8")

from app.agents.graph import build_graph  # noqa: E402
from app.agents.state import AgentState  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm import groq_client  # noqa: E402
from app.models.schemas import RefusalReason, is_insufficient_context  # noqa: E402
from app.observability.tracing import TraceRecorder  # noqa: E402
from judge_prompts import (  # noqa: E402
    COMPLETENESS_PROMPT,
    FAITHFULNESS_PROMPT,
    JUDGE_MODEL,
)

GATE_FAITHFULNESS = 0.75
GATE_REFUSAL = 2 / 3

# The harness is a batch job. Nobody is waiting on any single answer, so it
# does NOT inherit settings.deadline_seconds, which exists to stop a person
# staring at a spinner. Applying an interactive deadline here makes the eval
# report a provider rate limit as a quality regression — the one thing a gate
# must never do.
EVAL_DEADLINE_SECONDS = 600

# Groq's free tier allows 8,000 tokens per minute. One question costs roughly
# 1,500 across synthesis and grading, so firing twenty back to back is ~30,000
# tokens inside a minute: the harness rate-limits ITSELF, and then reports the
# damage as a drop in answer quality. Measured doing exactly that — 14 of 17
# answerable questions "refused" in under 120 ms each, at a mean of 339
# tokens/query against a normal 1,482.
#
# Pace to stay under the cap. A slow, trustworthy gate beats a fast one nobody
# believes. Override with EVAL_PACING_SECONDS=0 when running against a paid
# tier or a local model.
EVAL_PACING_SECONDS = float(os.environ.get("EVAL_PACING_SECONDS", "12"))

# Chunk ids gained a page segment ("doc_p2_s1_c0") when PDF extraction became
# page-aware. The golden dataset records which SECTION should be retrieved, not
# which page it happened to land on, so the page component is normalized away
# before comparing. Without this the hit-rate silently reads 0% after any id
# format change — a metric that fails quietly is worse than no metric.
_PAGE_SEGMENT_RE = re.compile(r"_p\d+(?=_s\d+_c\d+$)")


def _normalize_chunk_id(chunk_id: str) -> str:
    return _PAGE_SEGMENT_RE.sub("", chunk_id)


def _judge(system_prompt: str, user_content: str, key: str) -> float:
    content, _, _ = groq_client.chat_completion(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=300,  # headroom for gpt-oss's hidden reasoning tokens, see groq_client.py
        json_mode=True,
        # Grading is the tail of a burst — the answer it grades has just spent
        # the token bucket — so it is the call most likely to meet a 429, and
        # the default three attempts with 1+2+4s of backoff is not enough time
        # for a per-minute bucket to refill. Nobody is waiting on a batch job,
        # so wait properly. The deadline keeps "wait" bounded.
        max_retries=8,
        deadline_ts=time.monotonic() + EVAL_DEADLINE_SECONDS,
    )
    return float(json.loads(content)[key])


def judge_faithfulness(question: str, context: str, candidate: str) -> float:
    """Are the candidate's claims supported by the evidence actually retrieved?

    Judged against CONTEXT, not against the reference answer. Judging against
    the reference penalized true, document-supported detail that the terse
    reference happened to omit — worked examples in judge_prompts.py.
    """
    return _judge(
        FAITHFULNESS_PROMPT,
        f"Question: {question}\n\nCONTEXT:\n{context}\n\nCandidate: {candidate}",
        "faithfulness",
    )


def judge_completeness(question: str, reference: str, candidate: str) -> float:
    """Does the answer actually contain what was asked for?

    The counterweight to judging faithfulness against context alone: without
    this, an answer could faithfully quote unrelated context and score 1.0
    while never addressing the question.
    """
    return _judge(
        COMPLETENESS_PROMPT,
        f"Question: {question}\nReference: {reference}\nCandidate: {candidate}",
        "completeness",
    )


def run_item(graph, item: dict) -> dict:
    settings = get_settings()
    trace = TraceRecorder(query_redacted=item["question"])
    state: AgentState = {
        "query": item["question"], "search_query": item["question"],
        "user_id": "eval", "doc_id": None, "trace": trace, "attempt": 0,
        "model": settings.groq_model_simple,
        "token_budget_left": settings.token_budget,
        # Batch deadline, not the interactive one — see EVAL_DEADLINE_SECONDS.
        "deadline_ts": time.monotonic() + EVAL_DEADLINE_SECONDS,
        "retrieved": [], "answer": "", "citations": [], "critic": None,
        "status": "running", "refusal_reason": None, "conversation_history": [],
    }
    t0 = time.perf_counter()
    try:
        result = graph.invoke(state)
    except Exception as exc:  # noqa: BLE001 — reported as "unmeasured", see main()
        # An upstream failure (Groq rate limit, circuit breaker, network) is NOT
        # a quality regression, and scoring it as one is worse than not scoring
        # it at all: a gate that goes red for reasons unrelated to the change
        # trains everyone to ignore it. Recorded distinctly so main() can say
        # "could not measure" instead of "faithfulness dropped".
        return {
            "id": item["id"], "category": item["category"], "error": f"{type(exc).__name__}: {exc}",
            "refused": False, "hit": None, "faithfulness": 0.0, "completeness": 0.0,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "tokens": trace.total_tokens(), "repairs": trace.repair_count, "answer": "",
        }
    latency_ms = int((time.perf_counter() - t0) * 1000)

    if result.get("refusal_reason") == RefusalReason.BUDGET_EXHAUSTED:
        # The graph ran out of wall clock or tokens. That is a statement about
        # the provider and the budget, NOT about whether the documents answer
        # the question — and scoring it as a false refusal is exactly the
        # "quality regression that did not happen" this file already refuses
        # to report for exceptions. The same reasoning has to cover refusals,
        # because the deadline now surfaces as a clean BUDGET_EXHAUSTED rather
        # than as a raised error.
        return {
            "id": item["id"], "category": item["category"],
            "error": "BUDGET_EXHAUSTED — provider rate limit or deadline, not a quality signal",
            "refused": False, "hit": None, "faithfulness": 0.0, "completeness": 0.0,
            "latency_ms": latency_ms, "tokens": trace.total_tokens(),
            "repairs": trace.repair_count, "answer": "",
        }

    # Same helper the API uses, so the harness and production agree on what
    # counts as a refusal. A strict `== "INSUFFICIENT_CONTEXT"` here would miss
    # a decorated sentinel that the API correctly treats as a refusal, and the
    # refusal-rate metric would silently disagree with the shipped behaviour.
    refused = result["status"] == "refused" or is_insufficient_context(result["answer"])
    retrieved_ids = {_normalize_chunk_id(c.chunk_id) for c in result["retrieved"]}
    expected = {_normalize_chunk_id(c) for c in item["expected_chunk_ids"]}
    hit = bool(expected & retrieved_ids) if expected else None

    context = "\n\n".join(c.text for c in result["retrieved"])
    if item["category"] == "unanswerable":
        faith = 1.0 if refused else 0.0  # refusing IS the correct answer here
        completeness = 1.0 if refused else 0.0
    elif refused:
        faith = 0.3
        completeness = 0.0  # a false refusal answers nothing
    else:
        try:
            faith = judge_faithfulness(item["question"], context, result["answer"])
            completeness = judge_completeness(
                item["question"], item["reference_answer"], result["answer"]
            )
        except Exception as exc:  # noqa: BLE001 — reported as "unmeasured"
            # The ANSWER succeeded; only the grading of it failed. That costs
            # one unmeasured question, never the run: an unhandled judge error
            # used to abort main() entirely, so a single 429 at the wrong
            # moment threw away nineteen perfectly good results and reported
            # nothing at all.
            return {
                "id": item["id"], "category": item["category"],
                "error": f"judge unavailable: {type(exc).__name__}: {exc}",
                "refused": False, "hit": hit, "faithfulness": 0.0, "completeness": 0.0,
                "latency_ms": latency_ms, "tokens": trace.total_tokens(),
                "repairs": trace.repair_count, "answer": result["answer"][:120],
            }

    return {
        "id": item["id"], "category": item["category"], "refused": refused,
        "hit": hit, "faithfulness": faith, "completeness": completeness,
        "latency_ms": latency_ms,
        "tokens": trace.total_tokens(), "repairs": trace.repair_count,
        "answer": result["answer"][:120],
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    items = json.loads((root / "golden_dataset.json").read_text(encoding="utf-8"))
    graph = build_graph()

    rows = []
    for n, item in enumerate(items):
        if n and EVAL_PACING_SECONDS:
            time.sleep(EVAL_PACING_SECONDS)  # stay under the tokens/minute cap
        row = run_item(graph, item)
        rows.append(row)
        if row.get("error"):
            print(f"  [{row['id']:>2}] {row['category']:<12} UNMEASURED — {row['error'][:80]}")
        else:
            print(f"  [{row['id']:>2}] {row['category']:<12} faith={row['faithfulness']:.2f} "
                  f"compl={row['completeness']:.2f} hit={row['hit']} "
                  f"refused={row['refused']} {row['latency_ms']}ms")

    # Items that never produced an answer are excluded from every quality
    # metric. Averaging a 0.0 from a rate-limited request into faithfulness
    # would report a quality regression that did not happen.
    unmeasured = [r for r in rows if r.get("error")]
    if unmeasured:
        print(f"\n!! {len(unmeasured)}/{len(rows)} questions could not be measured:")
        for r in unmeasured:
            print(f"   [{r['id']}] {r['error'][:120]}")
        print(
            "\nEVAL INCONCLUSIVE — these are upstream failures (rate limit, "
            "circuit breaker, network), not quality regressions. Re-run when "
            "the upstream is healthy.",
            file=sys.stderr,
        )
        sys.exit(2)  # distinct from 1 (a real gate failure)

    answerable = [r for r in rows if r["category"] != "unanswerable"]
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    mean_faith = statistics.mean(r["faithfulness"] for r in rows)
    mean_completeness = statistics.mean(r["completeness"] for r in rows)
    hits = [r["hit"] for r in answerable if r["hit"] is not None]
    hit_rate = sum(hits) / len(hits)
    refusal_rate = sum(r["refused"] for r in unanswerable) / len(unanswerable)
    false_refusals = sum(r["refused"] for r in answerable)
    latencies = sorted(r["latency_ms"] for r in rows)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]
    mean_tokens = statistics.mean(r["tokens"] for r in rows)
    repaired = [r for r in rows if r["repairs"] > 0]

    report = [
        "# Sentinel — Eval Report", "",
        f"_{time.strftime('%Y-%m-%d %H:%M')} · {len(rows)} questions · judge: {JUDGE_MODEL}_", "",
        "| Metric | Value | Gate |", "|---|---|---|",
        f"| Mean faithfulness (vs retrieved context) | **{mean_faith:.3f}** | "
        f">= {GATE_FAITHFULNESS} |",
        f"| Mean completeness (vs reference) | {mean_completeness:.3f} | not gated yet |",
        f"| Retrieval hit-rate (top-{get_settings().top_k}) | {hit_rate:.0%} | — |",
        f"| Unanswerable refusal-rate | {refusal_rate:.0%} "
        f"({sum(r['refused'] for r in unanswerable)}/{len(unanswerable)}) | >= 2/3 |",
        f"| False refusals (answerable) | {false_refusals}/{len(answerable)} | — |",
        f"| Latency p50 / p95 | {p50} ms / {p95} ms | — |",
        f"| Mean tokens/query | {mean_tokens:.0f} | — |", "",
        "## Repair loop",
        f"{len(repaired)} of {len(rows)} questions triggered repair "
        f"({', '.join(str(r['id']) for r in repaired) or 'none'}). "
        "Faithfulness on repaired items: "
        + (f"{statistics.mean(r['faithfulness'] for r in repaired):.2f}" if repaired else "n/a"),
        "", "## Per-question results", "",
        "| # | Category | Faith | Compl | Hit | Refused | Repairs | ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    report += [
        f"| {r['id']} | {r['category']} | {r['faithfulness']:.2f} "
        f"| {r['completeness']:.2f} | {r['hit']} "
        f"| {r['refused']} | {r['repairs']} | {r['latency_ms']} |"
        for r in rows
    ]
    (root / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nreport written: {root / 'report.md'}")
    print(f"mean_faithfulness={mean_faith:.3f}  mean_completeness={mean_completeness:.3f}  "
          f"hit_rate={hit_rate:.0%}  unanswerable_refusal={refusal_rate:.0%}")

    if mean_faith < GATE_FAITHFULNESS or refusal_rate < GATE_REFUSAL:
        print("EVAL GATES FAILED", file=sys.stderr)
        sys.exit(1)
    print("EVAL GATES PASSED")


if __name__ == "__main__":
    main()
