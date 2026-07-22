"""Judge prompts for evals — isolated so prompt changes are visible in diffs.

The judge model and prompts are pinned: scores are only comparable across
runs if the ruler doesn't change between measurements.
"""

JUDGE_MODEL = "llama-3.1-8b-instant"

FAITHFULNESS_PROMPT = (
    "You are a strict evaluation judge. Given a question and the reference answer, "
    "and a candidate answer, score the candidate 0.0-1.0 on faithfulness: are "
    "its factual claims consistent with the reference (1.0 = fully consistent, "
    "0.0 = contradicts or fabricates)? A refusal/INSUFFICIENT_CONTEXT when the "
    "reference shows the question IS answerable scores 0.3. Output ONLY JSON: "
    '{"faithfulness": <float>, "why": "<one sentence>"}'
)
