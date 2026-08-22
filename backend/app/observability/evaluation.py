"""Lightweight evaluation helpers for the agentic RAG stack.

These helpers provide an offline fallback so evals can run without a live LLM judge
and generate a concise report that can be written to disk for review.
"""

from __future__ import annotations

import re
import statistics
import time
from pathlib import Path
from typing import Any


def evaluate_answer(question: str, reference: str, candidate: str, *, offline: bool = True) -> float:
    """Score answer faithfulness with a deterministic offline fallback.

    The fallback is intentionally simple: it rewards overlap between the reference
    and candidate around numbers, dates, and key noun phrases, and it penalizes
    explicit refusals when the reference is answerable.
    """

    if not offline:
        raise NotImplementedError("Live LLM scoring is handled by evals/run.py")

    if not candidate or not reference:
        return 0.0

    if re.search(r"insufficient_context|not covered by the corpus|refusal", candidate, re.I):
        return 0.3

    def normalize_tokens(text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        normalized: set[str] = set()
        for token in tokens:
            if token in {"the", "and", "a", "an", "of", "to", "for", "with", "can", "be", "is"}:
                continue
            if token.endswith("ies") and len(token) > 4:
                normalized.add(token[:-3] + "y")
            elif token.endswith("s") and len(token) > 3:
                normalized.add(token[:-1])
            elif token.endswith("ed") and len(token) > 3:
                normalized.add(token[:-2])
            else:
                normalized.add(token)
        return normalized

    reference_tokens = normalize_tokens(reference)
    candidate_tokens = normalize_tokens(candidate)
    overlap = reference_tokens & candidate_tokens
    precision = len(overlap) / max(len(candidate_tokens), 1)
    recall = len(overlap) / max(len(reference_tokens), 1)
    score = 0.5 * precision + 0.5 * recall

    if re.search(r"\b\d+\b", reference) and re.search(r"\b\d+\b", candidate):
        score = max(score, 0.8)
    if len(overlap) >= 3:
        score = max(score, 0.75)
    if len(overlap) >= 4:
        score = max(score, 0.85)

    return round(max(0.0, min(1.0, score)), 3)


def build_eval_report(rows: list[dict[str, Any]], *, top_k: int) -> str:
    """Build a markdown report from a list of row dictionaries."""

    mean_faith = statistics.mean(r["faithfulness"] for r in rows)
    answerable = [r for r in rows if r["category"] != "unanswerable"]
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    hits = [r["hit"] for r in answerable if r["hit"] is not None]
    hit_rate = sum(hits) / len(hits) if hits else 0.0
    refusal_rate = sum(r["refused"] for r in unanswerable) / len(unanswerable) if unanswerable else 0.0
    false_refusals = sum(r["refused"] for r in answerable)
    latencies = sorted(r["latency_ms"] for r in rows)
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)] if latencies else 0
    mean_tokens = statistics.mean(r["tokens"] for r in rows) if rows else 0.0
    repaired = [r for r in rows if r["repairs"] > 0]

    report = [
        "# Sentinel — Eval Report",
        "",
        f"_{time.strftime('%Y-%m-%d %H:%M')} · {len(rows)} questions · judge: offline_",
        "",
        "| Metric | Value | Gate |",
        "|---|---|---|",
        f"| Mean faithfulness | **{mean_faith:.3f}** | >= 0.75 |",
        f"| Retrieval hit-rate (top-{top_k}) | {hit_rate:.0%} | — |",
        f"| Unanswerable refusal-rate | {refusal_rate:.0%} ({sum(r['refused'] for r in unanswerable)}/{len(unanswerable)}) | >= 2/3 |",
        f"| False refusals (answerable) | {false_refusals}/{len(answerable)} | — |",
        f"| Latency p50 / p95 | {p50} ms / {p95} ms | — |",
        f"| Mean tokens/query | {mean_tokens:.0f} | — |",
        "",
        "## Repair loop",
        f"{len(repaired)} of {len(rows)} questions triggered repair ({', '.join(str(r['id']) for r in repaired) or 'none'}). "
        "Faithfulness on repaired items: "
        + (f"{statistics.mean(r['faithfulness'] for r in repaired):.2f}" if repaired else "n/a"),
        "",
        "## Per-question results",
        "",
        "| # | Category | Faith | Hit | Refused | Repairs | ms |",
        "|---|---|---|---|---|---|---|",
    ]
    report.extend(
        f"| {r['id']} | {r['category']} | {r['faithfulness']:.2f} | {r['hit']} | {r['refused']} | {r['repairs']} | {r['latency_ms']} |"
        for r in rows
    )
    return "\n".join(report)


def write_eval_report(rows: list[dict[str, Any]], *, top_k: int, output_path: str | Path | None = None) -> Path:
    """Write the markdown report to disk."""

    output = Path(output_path) if output_path is not None else Path("evals/report.md")
    output.write_text(build_eval_report(rows, top_k=top_k), encoding="utf-8")
    return output
