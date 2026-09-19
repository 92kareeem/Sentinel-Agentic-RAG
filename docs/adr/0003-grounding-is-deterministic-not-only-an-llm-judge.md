# ADR 0003 — Grounding is deterministic, and deliberately weak

**Status:** Accepted · **Date:** 2026-09-05 (documents pre-existing design)

## Context

Sentinel's product claim is that answers are grounded in the user's documents.
Something has to enforce that claim. The obvious candidate is the LLM critic
already in the graph, which scores faithfulness and relevance.

The problem: the critic is itself an LLM. It can be wrong in exactly the same
direction as the synthesizer — a fluent, plausible answer tends to *read* as
faithful, and asking a language model whether a language model told the truth
inherits the failure mode it is supposed to catch. An LLM-only guarantee is a
guarantee that fails silently and correlates with the error it is checking.

## Decision

Keep the critic, and put a **deterministic gate after it**
(`guardrails/grounding.py`) that cannot hallucinate because it contains no
model. For each cited sentence:

1. **Citation validity** — the cited chunk id must be in what was actually
   retrieved. An id that was never retrieved is fabricated.
2. **Content floor** — at least 2 real words, so an answer of bare citation
   tags cannot pass as fully grounded.
3. **Numeric grounding** — every number in the sentence must appear literally
   somewhere in the retrieved text.
4. **Lexical support** — the claim's content words must overlap the cited
   chunk above `MIN_LEXICAL_OVERLAP` (0.25).

Sentences failing any check are stripped. If more than `MAX_STRIPPED_RATIO`
(0.50) fail, the answer is untrustworthy and routes like a critic failure.

## The deliberate weakness

`MIN_LEXICAL_OVERLAP = 0.25` is low, and that is the decision, not an
oversight.

A stricter deterministic checker cannot distinguish "unsupported" from
"correctly paraphrased." Raising the threshold starts stripping legitimate
sentences that restate evidence in different words — which is what a good
answer *does*. That converts correct answers into false refusals: trading a
rare, subtle failure for a frequent, obvious one, and directly attacking the
0/17 false-refusal number.

So the division of labour is explicit:

| Question | Answered by | Nature |
|---|---|---|
| Does this cite something real? | Grounding gate | Deterministic, exact |
| Do the numbers exist in evidence? | Grounding gate | Deterministic, exact |
| Is this claim about the same subject as its evidence? | Grounding gate | Deterministic, weak |
| Does the evidence *entail* the claim? | LLM critic | Probabilistic |

The deterministic layer catches claims with essentially *nothing* to do with
their evidence. Judging meaning stays the critic's job.

## Alternatives considered

**Critic only.** Rejected — no deterministic floor; a confident wrong answer
with a plausible citation passes.

**An NLI/entailment model.** Genuinely stronger for the entailment question,
and the natural upgrade path. Rejected for now: another model dependency,
another cold-start cost, and another thing to evaluate, at a stage where the
measured failure mode was not "subtly unentailed claims." Revisit when the
eval shows entailment errors that the current stack misses.

**Requiring exact quoted spans.** Rejected — it would forbid summarizing or
combining two chunks, which is most of the product's value.

## Consequences

**Guaranteed** (deterministic, no model in the path): a citation always points
at retrieved text; a number in the answer always appears in retrieved text.

**Not guaranteed:** semantic entailment. A sentence that shares vocabulary with
its chunk and invents no numbers can still misstate what the chunk means.

This limit is documented in `docs/SENTINEL_SYSTEM_GUIDE.md` §6.5 and in the
README rather than glossed over, because a system whose guarantees are
precisely scoped is more trustworthy than one whose guarantees are
overstated — and overstating this particular one would undermine the entire
product claim.

## Implementation notes worth preserving

Two details in the current implementation exist because their absence caused
real bugs:

- **Numbers are validated against the whole retrieved context, not just the
  cited chunk.** Per-sentence citation is imperfect; the model may cite chunk A
  for a figure living in chunk B, both retrieved. Checking the union avoids
  punishing a misattributed-but-true claim, while a number in *no* retrieved
  chunk is still caught.
- **Sentence splitting walks tag-to-tag** rather than matching a
  sentence-shaped regex. A pattern that excluded `.` from the sentence body
  broke on abbreviations, decimals, and back-to-back citations — and an
  unparsed citation was neither kept nor counted as stripped, so a
  content-free answer could read as 100% grounded.
