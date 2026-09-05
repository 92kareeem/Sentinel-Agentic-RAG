# ADR 0004 — Faithfulness is judged against retrieved context, not the reference answer

**Status:** Accepted · **Date:** 2026-09-05

## Context

Two golden questions had been scoring 0.5–0.7 faithfulness across every run.
The temptation with a persistent low score is to treat it as noise, or as a
known-weak case, and let the mean absorb it. Investigating it instead
(`evals/diagnose.py`) turned up something more serious than a bad answer.

### Question 11 — *"What is required for production database access?"*

Source document:

> System access follows least privilege. Requests go through the ACCESS-REQ
> form and require manager approval. Production database access additionally
> requires the on-call certification exam (code OPS-204) with a passing score
> of 80%.

Sentinel's answer:

> Production database access requires manager approval and passing the on-call
> certification exam (code OPS-204) with a score of at least 80%.
> `[chunk:onboarding-guide_p0_s1_c0]`

Judge: **0.7** — *"adds an extra manager approval that is not stated in the
reference."*

### Question 12 — *"How do system access requests work?"*

Sentinel's answer reproduced the passage essentially verbatim, including the
OPS-204 sentence.

Judge: **0.5** — *"adds an unsupported detail about a certification exam, which
is not mentioned in the reference."*

### The actual defect

**Both answers are correct.** Every clause appears in the source document. The
internal critic scored both `faithfulness=1.0, relevance=1.0`, and it was
right.

The judge was penalizing them because the old prompt asked whether the
candidate's claims were *"consistent with the reference answer"* — and the
reference answers are deliberately terse one-liners. Any additional true,
document-supported detail read to the judge as fabrication.

The system was being marked down for being more complete than the answer key.

## Why this mattered more than two low scores

An inaccurate metric is bad. A metric that **rewards the wrong behaviour** is
dangerous, because it points every future optimization in the wrong direction.

Under the old ruler, the way to raise the faithfulness score was to make the
system answer more tersely and hew closer to reference phrasing — abandoning
true, cited, document-supported content. That is a direct attack on the product
premise. Anyone tuning prompts against this number, in good faith, would have
made Sentinel worse and watched the metric improve.

It also means the historically reported 0.950–0.955 figures **understated**
real faithfulness.

## Decision

Split the measurement into the two distinct things that were conflated:

| Metric | Judged against | Question it answers |
|---|---|---|
| **Faithfulness** | Retrieved **context** | Is every claim supported by the evidence? |
| **Completeness** | Reference answer | Does it contain what was asked for? |

`judge_faithfulness()` now receives the retrieved chunks and is told
explicitly that extra detail present in the context is correct and must not be
penalized. `judge_completeness()` is new and keeps the reference answer as the
yardstick, where it genuinely belongs.

Completeness is **reported but not gated**. A gate needs a baseline across
several runs; inventing a threshold on first sight would be making up an SLA.

## Alternatives considered

**Expand the reference answers to include every supported fact.** Rejected.
It makes the dataset expensive to maintain, and it is still wrong in principle:
faithfulness is a property of the answer relative to its *evidence*, not
relative to one particular phrasing a human wrote.

**Drop the reference entirely and judge only against context.** Rejected — it
opens the opposite hole. An answer could faithfully quote unrelated retrieved
text, address nothing, and score 1.0. Completeness closes that.

**Accept the two low scores as known-weak cases.** Rejected. This is exactly
the failure the investigation was meant to prevent: the score was not measuring
answer quality at all, so "known-weak" would have been a false label on
correct behaviour.

## Consequences

⚠️ **The ruler changed. Reports written before 2026-09-05 are not comparable to
ones written after.** `judge_prompts.py` carries this warning at the top,
because "scores are only comparable across runs if the ruler doesn't change"
was the reason those prompts were pinned in the first place.

Note that this does **not** invalidate the before/after comparisons in
ADR 0001 (refusal short-circuit) and ADR 0002 (router): each of those used the
*old* judge consistently on both sides of its comparison, so they remain
apples-to-apples within themselves.

Two metrics now have to be read together — neither alone describes answer
quality. That is the honest shape of the thing being measured.

**Second-order lesson worth keeping:** the internal critic said 1.0 and the
external judge said 0.5, and *the critic was right*. A disagreement between two
evaluators is a signal to investigate, not to average.
