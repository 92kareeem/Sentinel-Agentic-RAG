from app.observability.evaluation import build_eval_report, evaluate_answer


def test_evaluate_answer_uses_offline_fallback_for_simple_matches() -> None:
    score = evaluate_answer(
        "How many days after purchase can a customer request a refund?",
        "Customers may request a refund within 30 days of purchase.",
        "Refunds can be requested within 30 days.",
        offline=True,
    )

    assert score >= 0.8


def test_build_eval_report_renders_summary_markdown() -> None:
    rows = [
        {"id": 1, "category": "factual", "faithfulness": 0.91, "hit": True, "refused": False, "repairs": 0, "latency_ms": 120, "tokens": 40},
        {"id": 2, "category": "unanswerable", "faithfulness": 1.0, "hit": None, "refused": True, "repairs": 0, "latency_ms": 95, "tokens": 35},
    ]

    report = build_eval_report(rows, top_k=4)

    assert "# Sentinel — Eval Report" in report
    assert "Mean faithfulness" in report
    assert "Unanswerable refusal-rate" in report
    assert "| 1 | factual | 0.91 | True | False | 0 | 120 |" in report
