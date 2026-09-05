"""Eval runner: golden dataset -> agent graph -> metrics -> evals/report.md.

Usage (from repo root, venv python):
    python evals/run.py            # full run against local index + live Groq

Exit code 1 if gates fail (mean faithfulness < 0.75 or unanswerable-refusal
< 2/3) so CI can use this directly.
"""

import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.agents.graph import build_graph  # noqa: E402
from app.agents.state import AgentState  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm import groq_client  # noqa: E402
from app.models.schemas import is_insufficient_context  # noqa: E402
from app.observability.tracing import TraceRecorder  # noqa: E402
from judge_prompts import (  # noqa: E402
    COMPLETENESS_PROMPT,
    FAITHFULNESS_PROMPT,
    JUDGE_MODEL,
)

GATE_FAITHFULNESS = 0.75
GATE_REFUSAL = 2 / 3

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
        "query": item["question"], "user_id": "eval", "doc_id": None, "trace": trace, "attempt": 0,
        "model": settings.groq_model_simple,
        "token_budget_left": settings.token_budget,
        "deadline_ts": time.monotonic() + settings.deadline_seconds,
        "retrieved": [], "answer": "", "citations": [], "critic": None,
        "status": "running", "refusal_reason": None, "conversation_history": [],
    }
    t0 = time.perf_counter()
    result = graph.invoke(state)
    latency_ms = int((time.perf_counter() - t0) * 1000)

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
        faith = judge_faithfulness(item["question"], context, result["answer"])
        completeness = judge_completeness(
            item["question"], item["reference_answer"], result["answer"]
        )

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
    for item in items:
        row = run_item(graph, item)
        rows.append(row)
        print(f"  [{row['id']:>2}] {row['category']:<12} faith={row['faithfulness']:.2f} "
              f"compl={row['completeness']:.2f} hit={row['hit']} "
              f"refused={row['refused']} {row['latency_ms']}ms")

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
