"""Judge prompts for evals — isolated so prompt changes are visible in diffs.

The judge model and prompts are pinned: scores are only comparable across runs
if the ruler doesn't change between measurements.

!! RULER CHANGED 2026-09-05 — see FAITHFULNESS_PROMPT below. Reports written
   before that date are NOT comparable to ones written after it.
"""

JUDGE_MODEL = "openai/gpt-oss-20b"  # llama-3.1-8b-instant was decommissioned by Groq

# ---------------------------------------------------------------------------
# Faithfulness: candidate vs RETRIEVED CONTEXT (not vs the reference answer).
#
# WHY THIS CHANGED
# ----------------
# The previous prompt scored the candidate against the terse reference answer
# and asked whether its claims were "consistent with the reference". That is
# not faithfulness, and it penalized correct behaviour. Two worked examples
# from the golden set (diagnosed with evals/diagnose.py):
#
#   Q11 "What is required for production database access?"
#     Answer: "...requires manager approval and passing the on-call
#              certification exam (code OPS-204) with a score of at least 80%."
#     Judge:  0.7 — "adds an extra manager approval that is not stated in the
#              reference."
#
#   Q12 "How do system access requests work?"
#     Answer: "...least privilege... ACCESS-REQ form... manager approval.
#              Production database access additionally requires the on-call
#              certification exam (OPS-204) with a passing score of 80%."
#     Judge:  0.5 — "adds an unsupported detail about a certification exam,
#              which is not mentioned in the reference."
#
# Both answers are fully correct and every clause appears verbatim in the
# source document. They were marked down for containing TRUE, document-supported
# information that the one-line reference happened to omit.
#
# That makes the metric actively harmful, not merely noisy: optimizing against
# it would push the system toward terse answers that mimic reference phrasing
# instead of complete answers grounded in the document. A metric that rewards
# the wrong behaviour is worse than no metric.
#
# Faithfulness in RAG means: is every claim supported by the evidence the
# system actually retrieved? So the judge is given the CONTEXT and scores
# against that. Whether the answer covers what was asked is a different
# question, measured separately by COMPLETENESS_PROMPT below.
# ---------------------------------------------------------------------------
FAITHFULNESS_PROMPT = (
    "You are a strict evaluation judge measuring FAITHFULNESS: is every factual "
    "claim in the candidate answer supported by the provided CONTEXT?\n"
    "\n"
    "Score 0.0-1.0:\n"
    "  1.0 = every claim is supported by the context\n"
    "  0.5 = some claims are supported, others are not in the context\n"
    "  0.0 = the answer contradicts the context or fabricates facts\n"
    "\n"
    "IMPORTANT: judge ONLY against the CONTEXT. Additional true detail that "
    "appears in the context is CORRECT and must NOT be penalized, even if it is "
    "absent from the reference answer — the reference is a minimal example, not "
    "an exhaustive list of what may be said. Ignore citation tags like "
    "[chunk:abc] when judging; they are formatting, not claims.\n"
    "\n"
    "A refusal (INSUFFICIENT_CONTEXT) when the context DOES support an answer "
    "scores 0.3.\n"
    "\n"
    'Output ONLY JSON: {"faithfulness": <float>, "why": "<one sentence>"}'
)

# ---------------------------------------------------------------------------
# Completeness: does the answer actually contain what was asked for?
#
# This is the counterweight to the faithfulness fix above. Judging faithfulness
# purely against context means an answer could quote unrelated-but-real context
# and score 1.0 while never answering the question. This catches that, and it
# is the metric for which the reference answer IS the right yardstick.
# ---------------------------------------------------------------------------
COMPLETENESS_PROMPT = (
    "You are a strict evaluation judge measuring COMPLETENESS: does the "
    "candidate answer contain the key facts present in the reference answer?\n"
    "\n"
    "Score 0.0-1.0:\n"
    "  1.0 = every key fact in the reference appears in the candidate\n"
    "  0.5 = roughly half of the reference's key facts appear\n"
    "  0.0 = the candidate does not answer the question at all\n"
    "\n"
    "Extra correct detail beyond the reference is fine and must NOT reduce the "
    "score. Wording need not match; judge the facts. Ignore citation tags.\n"
    "\n"
    'Output ONLY JSON: {"completeness": <float>, "why": "<one sentence>"}'
)
