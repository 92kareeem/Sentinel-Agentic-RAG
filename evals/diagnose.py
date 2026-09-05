"""Inspect what the system actually answers for specific golden questions.

The eval report gives a score; it does not tell you WHY a question scored what
it did. This runs chosen questions through the real graph and prints the
retrieved evidence, the final answer, and the judge's own reasoning, so a low
score can be diagnosed instead of averaged away.

Usage:  python evals/diagnose.py 11 12
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

# LLM output routinely contains characters the Windows console default (cp1252)
# cannot encode — a non-breaking hyphen in "on-call" was enough to crash this
# script mid-report. Diagnostics must survive the text they are diagnosing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.agents.graph import build_graph  # noqa: E402
from app.agents.state import AgentState  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm import groq_client  # noqa: E402
from app.observability.tracing import TraceRecorder  # noqa: E402
from judge_prompts import FAITHFULNESS_PROMPT, JUDGE_MODEL  # noqa: E402


def judge_verbose(question: str, reference: str, candidate: str) -> dict:
    content, _, _ = groq_client.chat_completion(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": FAITHFULNESS_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\nReference: {reference}\nCandidate: {candidate}",
            },
        ],
        max_tokens=300,
        json_mode=True,
    )
    return json.loads(content)


def main() -> None:
    wanted = {int(a) for a in sys.argv[1:]} or {11, 12}
    root = Path(__file__).resolve().parent
    items = json.loads((root / "golden_dataset.json").read_text(encoding="utf-8"))
    graph = build_graph()
    settings = get_settings()

    for item in items:
        if item["id"] not in wanted:
            continue

        trace = TraceRecorder(query_redacted=item["question"])
        state: AgentState = {
            "query": item["question"], "user_id": "eval", "doc_id": None, "trace": trace,
            "attempt": 0, "model": settings.groq_model_simple,
            "token_budget_left": settings.token_budget,
            "deadline_ts": time.monotonic() + settings.deadline_seconds,
            "retrieved": [], "answer": "", "citations": [], "critic": None,
            "status": "running", "refusal_reason": None, "conversation_history": [],
        }
        result = graph.invoke(state)

        print("=" * 78)
        print(f"[{item['id']}] {item['category']}: {item['question']}")
        print("=" * 78)
        print(f"\nREFERENCE:\n  {item['reference_answer']}")
        print(f"\nRETRIEVED ({len(result['retrieved'])} chunks):")
        for c in result["retrieved"]:
            print(f"  - {c.chunk_id} | {c.section_path}")
            print(f"      {c.text[:180].replace(chr(10), ' ')}")
        print(f"\nANSWER:\n  {result['answer']}")
        print(f"\nSTATUS: {result['status']}   critic: {result['critic']}")
        print(f"repairs: {trace.repair_count}   model_path: {trace.model_path}")

        if item["category"] != "unanswerable":
            verdict = judge_verbose(
                item["question"], item["reference_answer"], result["answer"]
            )
            print(f"\nJUDGE: {json.dumps(verdict, indent=2)}")
        print()


if __name__ == "__main__":
    main()
