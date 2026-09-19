# ADR 0006 — The search query is not the user's question

**Status:** Accepted · **Date:** 2026-09-19

## Context

`AgentState` had one field, `query`, used for four different jobs:

| Reader | What it actually needs |
|---|---|
| `retriever` | terms that match document vocabulary |
| `synthesizer` | the question to answer |
| `critic` | the question to score relevance against |
| `repair_rewrite` | the question to reformulate |

Those are not the same string, and `repair_rewrite` overwrote the field:

```python
state["query"] = content.strip() or state["query"]
```

The rewrite prompt asks for a reformulation that is "more specific and
retrieval-friendly, using terms likely to appear in source documents". That is
the right objective for *search* and the wrong one for *answering*: it drops
qualifiers, because qualifiers are the part least likely to appear verbatim in
a document.

Worked example, from the domain this product targets:

> **User asks:** "Can we refund an annual subscription after 20 days if
> onboarding has already started?"
>
> **Rewrite:** "annual subscription refund window policy"
>
> **Retrieval:** correctly finds the refund-window section.
>
> **Synthesizer:** answers the *rewritten* question — "Annual subscriptions may
> be refunded within 30 days. [chunk:refunds]"
>
> **Critic:** scores relevance against the rewritten question. High.
>
> **Grounding gate:** the claim matches its cited passage. Passes.

Every stage agrees. The citation is correct. The answer is a correct statement
about the refund window — and it silently drops the onboarding exception, which
is the entire thing the user asked about.

This is the worst failure shape this system can produce, because unlike a
hallucination there is nothing to detect. No claim is unsupported. The defect
is that a *different question* was answered, and no gate in the graph was
looking at the original question to notice.

## Decision

Split the field by role:

- **`query`** — what the user asked. Set once in `routes_query` and never
  reassigned by any node.
- **`search_query`** — what retrieval is currently searching with. Starts equal
  to `query`; `repair_rewrite` replaces it.

`retriever` reads `search_query`. `synthesizer` and `critic` read `query`. The
answer the user reads addresses what they asked, whatever retrieval did to find
the evidence.

`repair_rewrite` always reformulates **from `query`**, not from the previous
`search_query`, so drift cannot compound across attempts.

## Alternatives considered

**Feed both to the synthesizer.** Rejected. It spends context on a string the
model does not need, and inviting a model to reconcile two versions of a
question is a new failure mode where there was none.

**Stop rewriting.** Rejected — measurably useful. A rewrite re-runs retrieval,
and "the evidence isn't here" genuinely does sometimes mean "retrieval missed
it" (see ADR 0001, which keeps exactly one rewrite round for that reason). The
bug was never the rewrite; it was letting a retrieval-tuned string escape into
the answer path.

**Make `search_query` optional with a `.get()` fallback.** Rejected. A required
field forces every construction site to decide what it means, and mypy finds
the ones that do not. The looser version would have silently defaulted a future
caller into the old behaviour.

## Consequences

- One existing test asserted `result["query"] == "rewritten query"` — it was
  pinning the bug in place as if it were the contract. It now asserts both
  fields.
- The trace records the rewrite under `search_query`, so a reader can see both
  what was asked and what was searched. That difference is itself a useful
  diagnostic and feeds the casebook.
- Any future query transformation (conversational reference resolution,
  sub-question decomposition) has an obvious place to live, and an obvious
  constraint: it may touch `search_query`, never `query`.

## Related

- `backend/app/agents/state.py`
- `backend/app/agents/repair.py`
- `backend/tests/test_answer_integrity.py`
- ADR 0001 (refusal is not a repairable failure)
