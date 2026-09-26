"""Eval runner: golden dataset + regression suite -> agent graph -> evals/report.md.

Usage (from repo root, venv python):
    python evals/run.py                     # golden set + regressions, live Groq
    python evals/run.py --regressions-only  # just the ratchet, e.g. after a fix
    python evals/run.py --lock              # also lock pending regressions that now pass

Two suites, two kinds of gate:

  golden_dataset.json  AVERAGES: mean faithfulness >= 0.75, unanswerable
                       refusal >= 2/3. "Is the system good overall?"
  regressions.json     PER QUESTION: every enforced regression must pass.
                       "Did a mistake we already fixed come back?"

An average cannot answer the second question — one item can fall from 1.0
to 0.0 and move a 20-item mean by only 0.05 — which is why the ratchet is a
separate suite rather than more golden rows. See app/learning/ratchet.py.

Exit codes: 0 pass · 1 a gate failed · 2 inconclusive (upstream failures,
not quality regressions — re-run).
"""

import argparse
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
from app.learning import ratchet  # noqa: E402
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

# Groq's free tier allows 8,000 tokens per minute, refilled continuously.
#
# The harness used to fire the whole suite flat out — ~30,000 tokens inside a
# minute — and then score the resulting 429s as bad answers: 14 of 17
# answerable questions "refused" in under 120 ms each, at a mean of 339
# tokens/query against a normal 1,482.
#
# Pacing is derived from what each question ACTUALLY cost rather than from a
# guessed constant, because the cost varies by a factor of two: a question
# that escalates to the larger model runs synthesis twice. A fixed 12s sleep
# was tuned against the cheap case and still hit the limit on the expensive
# one. Spend N tokens, then wait for the bucket to refill N tokens.
#
# The 0.8 factor leaves headroom for the request this process is not the only
# one making — a developer's laptop and CI can share an account.
TOKENS_PER_MINUTE = 8_000
PACING_SAFETY = 0.8
_TOKENS_PER_SECOND = (TOKENS_PER_MINUTE * PACING_SAFETY) / 60

# Set to 0 on a paid tier or against a local model, where none of this applies.
EVAL_PACING = os.environ.get("EVAL_PACING", "1") != "0"


def _pace_for(tokens_spent: int) -> float:
    """Seconds to wait for the token bucket to refill what we just spent."""
    if not EVAL_PACING or tokens_spent <= 0:
        return 0.0
    return tokens_spent / _TOKENS_PER_SECOND


# Grading runs through groq_client too, and its tokens come out of the same
# bucket — roughly half of each question's real cost. They are tracked
# separately from the trace so the report's "tokens/query" keeps meaning what
# it says (what the SYSTEM spent answering), while pacing can account for what
# the whole harness spent.
_judge_tokens = 0

# Chunk ids gained a page segment ("doc_p2_s1_c0") when PDF extraction became
# page-aware. The golden dataset records which SECTION should be retrieved, not
# which page it happened to land on, so the page component is normalized away
# before comparing. Without this the hit-rate silently reads 0% after any id
# format change — a metric that fails quietly is worse than no metric.
_PAGE_SEGMENT_RE = re.compile(r"_p\d+(?=_s\d+_c\d+$)")


def _normalize_chunk_id(chunk_id: str) -> str:
    return _PAGE_SEGMENT_RE.sub("", chunk_id)


def _judge(system_prompt: str, user_content: str, key: str) -> float:
    global _judge_tokens
    content, tokens_in, tokens_out = groq_client.chat_completion(
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
    _judge_tokens += tokens_in + tokens_out
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
    global _judge_tokens
    _judge_tokens = 0
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
            "judge_tokens": _judge_tokens,
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
            "judge_tokens": _judge_tokens,
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
                "judge_tokens": _judge_tokens,
            }

    return {
        "id": item["id"], "category": item["category"], "refused": refused,
        "hit": hit, "faithfulness": faith, "completeness": completeness,
        "latency_ms": latency_ms,
        "tokens": trace.total_tokens(), "repairs": trace.repair_count,
        "answer": result["answer"][:120], "judge_tokens": _judge_tokens,
    }


def _as_eval_item(reg: ratchet.RegressionItem) -> dict:
    """A regression in the shape run_item() takes. `refuse` reuses the
    unanswerable path (refusing IS the pass); `answer` is judged against the
    promoted reference, which is where a reader's reviewed correction lands."""
    return {
        "id": reg.id,
        "category": "unanswerable" if reg.expect == "refuse" else "regression",
        "question": reg.question,
        "expected_chunk_ids": [],
        "reference_answer": reg.reference_answer or "",
    }


def _print_row(row: dict) -> None:
    if row.get("error"):
        print(f"  [{row['id']:>4}] {row['category']:<12} UNMEASURED — {row['error'][:80]}")
    else:
        print(f"  [{row['id']:>4}] {row['category']:<12} faith={row['faithfulness']:.2f} "
              f"compl={row['completeness']:.2f} hit={row['hit']} "
              f"refused={row['refused']} {row['latency_ms']}ms")


_prev_cost = 0  # tokens the previous run spent, answering and grading


def _run(graph, item: dict) -> dict:
    """run_item(), paced by what the PREVIOUS run actually cost.

    Waiting before a run rather than after one means the first run never
    waits and the last never leaves a pointless trailing sleep, however the
    runs are sequenced — golden, then regressions, then any confirmation
    re-runs. The wait is sized by measured cost, not a guess: a question that
    escalates spends about twice what a simple one does.
    """
    global _prev_cost
    wait = _pace_for(_prev_cost)
    if wait:
        print(f"       ...pacing {wait:.0f}s for the token bucket")
        time.sleep(wait)
    row = run_item(graph, item)
    _prev_cost = row["tokens"] + row.get("judge_tokens", 0)
    _print_row(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentinel eval runner")
    parser.add_argument(
        "--regressions-only", action="store_true",
        help="run only the regression suite (skips the golden-set averages)",
    )
    parser.add_argument(
        "--lock", action="store_true",
        help="lock pending regressions that pass in this run (writes regressions.json)",
    )
    parser.add_argument(
        "--suite", type=Path, default=None,
        help="regression suite file (default: evals/regressions.json)",
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help="where to write the report (default: evals/report.md for a full run; "
             "--regressions-only prints instead of writing unless this is given)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    suite_path = args.suite or root / "regressions.json"
    suite = ratchet.load(suite_path)
    # A hand-edited suite that breaks an invariant (a deleted regression, an
    # answer test with no reference) fails here, before spending any quota.
    problems = ratchet.validate(suite)
    if problems:
        print("regressions.json is invalid:\n  " + "\n  ".join(problems), file=sys.stderr)
        sys.exit(1)

    items = (
        [] if args.regressions_only
        else json.loads((root / "golden_dataset.json").read_text(encoding="utf-8"))
    )
    graph = build_graph()

    rows = []
    for item in items:
        rows.append(_run(graph, item))

    # --- the ratchet
    if suite.active:
        print(f"\n  regression suite: {len(suite.active)} active "
              f"({sum(i.status == 'enforced' for i in suite.active)} enforced)")
    outcomes: list[ratchet.Outcome] = []
    reg_rows: list[dict] = []
    for reg in suite.active:
        row = _run(graph, _as_eval_item(reg))
        outcome = ratchet.evaluate(reg, row)
        if outcome.status == "enforced" and outcome.passed is False:
            # Confirm before failing the build. One judge call is noisy, and a
            # ratchet that goes red on noise gets switched off — which would
            # undo everything it exists to protect. A real regression fails
            # twice; a wobble usually does not.
            print(f"         {reg.id} failed ({outcome.reason}); confirming with one re-run")
            row = _run(graph, _as_eval_item(reg))
            retry = ratchet.evaluate(reg, row)
            if retry.passed is not False:
                outcome = retry.model_copy(
                    update={"reason": f"passed on re-run (first run: {outcome.reason})"}
                )
            else:
                outcome = retry.model_copy(update={"reason": f"{retry.reason} (confirmed)"})
        outcomes.append(outcome)
        reg_rows.append(row)

    # Items that never produced an answer are excluded from every quality
    # metric. Averaging a 0.0 from a rate-limited request into faithfulness
    # would report a quality regression that did not happen.
    unmeasured = [r for r in rows + reg_rows if r.get("error")]
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

    blocking = ratchet.blocking_failures(outcomes)
    lockable = ratchet.ready_to_lock(outcomes)
    regression_report = _regression_section(suite, outcomes)

    if args.lock and lockable:
        for o in lockable:
            ratchet.lock(suite, o.id)
        ratchet.save(suite, suite_path)
        print(f"\nlocked: {', '.join(o.id for o in lockable)} — commit evals/regressions.json")
        lockable = []

    if not rows:  # --regressions-only
        # Never onto evals/report.md by default. That file is the committed
        # record of the last FULL run; a partial run written over it would
        # erase the golden-set numbers while you were only re-checking one
        # fix — found by doing exactly that during end-to-end testing.
        text = "# Sentinel — Regression Report\n\n" + "\n".join(regression_report)
        if args.report:
            args.report.write_text(text, encoding="utf-8")
            print(f"\nreport written: {args.report}")
        else:
            print("\n" + text)
        _finish(blocking, lockable, aggregate_ok=True)
        return

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
    report += ["", *regression_report]
    report_path = args.report or root / "report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nreport written: {report_path}")
    print(f"mean_faithfulness={mean_faith:.3f}  mean_completeness={mean_completeness:.3f}  "
          f"hit_rate={hit_rate:.0%}  unanswerable_refusal={refusal_rate:.0%}")

    _finish(
        blocking, lockable,
        aggregate_ok=mean_faith >= GATE_FAITHFULNESS and refusal_rate >= GATE_REFUSAL,
    )


def _regression_section(
    suite: ratchet.RegressionSuite, outcomes: list[ratchet.Outcome]
) -> list[str]:
    enforced = sum(i.status == "enforced" for i in suite.active)
    lines = [
        "## Regression suite",
        "",
        f"{enforced} enforced · {len(suite.active) - enforced} pending · "
        f"{len(suite.retired)} retired. Every enforced item must pass on its own; "
        "pending items are known failures, tracked but not blocking.",
    ]
    if not outcomes:
        return [*lines, "", "_No active regressions yet._"]
    by_id = {i.id: i for i in suite.active}
    lines += ["", "| ID | Status | Result | Why | From |", "|---|---|---|---|---|"]
    for o in outcomes:
        item = by_id[o.id]
        result = {True: "pass", False: "**FAIL**", None: "unmeasured"}[o.passed]
        lines.append(
            f"| {o.id} | {o.status} | {result} | {o.reason} "
            f"| {item.source.diagnosis.value}, asked {item.source.reported_at[:10]} |"
        )
    return lines


def _finish(
    blocking: list[ratchet.Outcome], lockable: list[ratchet.Outcome], *, aggregate_ok: bool
) -> None:
    if lockable:
        print(
            f"\n{len(lockable)} pending regression(s) now pass: "
            f"{', '.join(o.id for o in lockable)}. Lock them so they can never fail "
            "unnoticed again:  python evals/run.py --lock   (or promote.py lock <id>)"
        )
    if blocking:
        print("\nREGRESSIONS — mistakes that were fixed have come back:", file=sys.stderr)
        for o in blocking:
            print(f"   {o.id}: {o.reason}", file=sys.stderr)
    if not aggregate_ok or blocking:
        print("EVAL GATES FAILED", file=sys.stderr)
        sys.exit(1)
    print("EVAL GATES PASSED")


if __name__ == "__main__":
    main()
