"""Does the LLM router earn its cost? Measure before deciding.

The router spends one LLM call on EVERY query to classify it as
simple / multi_hop / needs_table, which selects the 20b or 120b model. A
keyword heuristic already exists in router.py as the fallback path. That makes
the question empirical, not architectural:

  1. How often does the LLM label differ from the heuristic label?
  2. Of those disagreements, how often does the choice of MODEL actually change?
     (Only "simple" vs "not simple" matters — multi_hop and needs_table both
     select the same complex model, so disagreement between those two is free.)
  3. What does the call cost in latency and tokens?

A router that rarely changes the model is pure overhead on every request.

Usage (from repo root):  python evals/router_ab.py
Writes evals/router_report.md. Read-only with respect to the index; makes one
small LLM call per question.
"""

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.agents.router import _heuristic  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm import groq_client  # noqa: E402

# The router no longer makes this call (ADR 0002) — this harness keeps it so
# the decision can be re-measured rather than re-argued. Sized for gpt-oss's
# hidden reasoning tokens: at 20 the visible content comes back empty.
_CLASSIFY_MAX_TOKENS = 120

_CLASSIFY_PROMPT = (
    "Classify the user question as exactly one word: "
    "simple, multi_hop, or needs_table. No explanation."
)


def llm_label(question: str) -> tuple[str, int, int, int]:
    """Returns (label, tokens_in, tokens_out, latency_ms) — the same call
    router_node makes, so the numbers are the real cost, not an estimate."""
    settings = get_settings()
    t0 = time.perf_counter()
    content, tin, tout = groq_client.chat_completion(
        model=settings.groq_model_simple,
        messages=[
            {"role": "system", "content": _CLASSIFY_PROMPT},
            {"role": "user", "content": question},
        ],
        max_tokens=_CLASSIFY_MAX_TOKENS,
    )
    ms = int((time.perf_counter() - t0) * 1000)
    cleaned = content.strip().lower()
    if cleaned not in {"simple", "multi_hop", "needs_table"}:
        cleaned = "UNPARSEABLE"
    return cleaned, tin, tout, ms


def model_for(label: str) -> str:
    s = get_settings()
    return s.groq_model_simple if label == "simple" else s.groq_model_complex


def main() -> None:
    root = Path(__file__).resolve().parent
    items = json.loads((root / "golden_dataset.json").read_text(encoding="utf-8"))

    rows = []
    for item in items:
        q = item["question"]
        heur = _heuristic(q)
        llm, tin, tout, ms = llm_label(q)
        rows.append({
            "id": item["id"],
            "question": q,
            "heuristic": heur,
            "llm": llm,
            "label_differs": heur != llm,
            "model_differs": model_for(heur) != model_for(llm),
            "tokens": tin + tout,
            "ms": ms,
        })
        print(f"  [{item['id']:>2}] heuristic={heur:<11} llm={llm:<12} "
              f"model_changed={rows[-1]['model_differs']}  {ms}ms")

    n = len(rows)
    label_diff = sum(r["label_differs"] for r in rows)
    model_diff = sum(r["model_differs"] for r in rows)
    unparseable = sum(r["llm"] == "UNPARSEABLE" for r in rows)
    mean_ms = statistics.mean(r["ms"] for r in rows)
    total_ms = sum(r["ms"] for r in rows)
    mean_tokens = statistics.mean(r["tokens"] for r in rows)

    report = [
        "# Router A/B — is the classification LLM call worth it?", "",
        f"_{time.strftime('%Y-%m-%d %H:%M')} · {n} questions · "
        f"classifier: {get_settings().groq_model_simple}_", "",
        "The router spends one LLM call per query to pick between the 20b and",
        "120b synthesis model. router.py already contains a keyword heuristic",
        "used whenever that call fails. This compares the two.", "",
        "| Metric | Value |", "|---|---|",
        f"| Questions | {n} |",
        f"| Label disagreements (LLM vs heuristic) | {label_diff}/{n} ({label_diff/n:.0%}) |",
        f"| **Disagreements that change the MODEL** | **{model_diff}/{n} ({model_diff/n:.0%})** |",
        f"| Unparseable LLM labels | {unparseable}/{n} |",
        f"| Mean router latency | {mean_ms:.0f} ms |",
        f"| Total router latency across the set | {total_ms} ms |",
        f"| Mean router tokens | {mean_tokens:.0f} |", "",
        "Only the model-changing row matters for outcomes: `multi_hop` and",
        "`needs_table` both select the complex model, so disagreement between",
        "those two costs nothing. Every query pays the latency row regardless.",
        "", "## Per-question", "",
        "| # | Heuristic | LLM | Model changed | ms | Question |",
        "|---|---|---|---|---|---|",
    ]
    report += [
        f"| {r['id']} | {r['heuristic']} | {r['llm']} | "
        f"{'**yes**' if r['model_differs'] else 'no'} | {r['ms']} | {r['question'][:60]} |"
        for r in rows
    ]
    (root / "router_report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"\nreport written: {root / 'router_report.md'}")
    print(f"label_disagreement={label_diff}/{n}  model_changed={model_diff}/{n}  "
          f"mean_router_latency={mean_ms:.0f}ms")


if __name__ == "__main__":
    main()
