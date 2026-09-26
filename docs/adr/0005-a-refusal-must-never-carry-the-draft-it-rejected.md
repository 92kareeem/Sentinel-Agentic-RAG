# ADR 0005 — A refusal must never carry the draft it rejected

**Status:** Accepted · **Date:** 2026-09-19

## Context

Every refusal in the graph funnels through `refusal_node`, which sets
`status="refused"` and a `RefusalReason`. The API then returned the refusal as:

```python
RefusalResponse(reason=result["answer"], ...)
```

For the honest case — the synthesizer emitting `INSUFFICIENT_CONTEXT` — that is
harmless. For the other two it is not. When the critic or the grounding gate
rejects an answer, `state["answer"]` still holds **the rejected draft**, and
`refusal_node` only replaced it when it was empty:

```python
if not state.get("answer"):
    state["answer"] = "I don't have enough verified context..."
```

So the one piece of text the system had just decided it could not stand behind
was returned to the caller, labelled as the explanation for withholding it. The
refusal functioned as a disclaimer in front of the unverified answer rather
than as a gate in place of it.

Two things made this hard to notice:

- The **UI never showed it.** The frontend renders its own copy keyed on
  `reason_code`, so a browser demo looks correct no matter what `reason` holds.
  Only an API consumer — an integration, a script, a partner — sees the leak.
- Every test asserted on `reason_code`, which was right, so nothing failed.

This is the most damaging possible bug for a product whose entire claim is that
every statement is supported by the user's documents, because the failure is
*silent* and *inverted*: the more suspicious the draft, the more certain it is
to be shown.

## Decision

**Refusal prose is generated from `RefusalReason`, never from model output.**

`safe_refusal_text()` lives in `models/schemas.py`, next to the enum it maps
from, because it is part of the API contract rather than agent behaviour.

Two layers apply it independently:

1. `refusal_node` overwrites `state["answer"]` with the safe text and clears
   `state["citations"]`.
2. `routes_query` derives `reason` from `reason_code` and **never reads
   `result["answer"]`** on the refusal path.

Layer 2 is not redundant. Layer 1 is a property of one node that future work
could weaken; layer 2 makes the leak unreachable from the API regardless of
what any node leaves in state. The invariant we want is "no model text reaches
`RefusalResponse.reason`", and that is enforced where the response is built.

The rejected draft is **kept in state** as `rejected_draft` for triage. It is
the clearest available signal for separating a retrieval miss from a synthesis
failure — what the model *wanted* to say tells you whether the evidence was
there — and the casebook consumes it. It never enters a response body.

Citations are dropped on refusal because they described claims that are no
longer being made; leaving them would let a client render sources beneath a
refusal and imply the refusal itself was evidence-backed.

## Alternatives considered

**Keep the draft but label it "unverified".** Rejected. There is no framing
that makes it safe — the text is on screen, it is fluent, and a reader takes
away the claim rather than the caveat. "Unverified" is also precisely what the
system could not determine; the honest statement is that we do not know, and
the draft adds nothing to that.

**Return the draft only for `UNVERIFIABLE_ANSWER`, where evidence existed.**
Rejected. That is the case where the draft is *most* likely to be a
confident-sounding hallucination: evidence was retrieved, an answer was
written, and verification failed. Evidence existing makes a wrong answer more
plausible, not more permissible.

**Blank the draft in `refusal_node` only.** Rejected as insufficient — see
layer 2 above.

## Consequences

- `RefusalResponse.reason` is now one of three fixed strings. Clients that
  pattern-matched on prose were already told to switch to `reason_code`
  (ADR-era change), and the fixed set is easier to match on than free text.
- Triage keeps full fidelity: the draft is available on the state and via the
  `refusal` trace step.
- A future node cannot reintroduce the leak by leaving model output in
  `answer`.

## Related

- `backend/app/agents/graph.py` — `refusal_node`
- `backend/app/models/schemas.py` — `RefusalReason`, `safe_refusal_text`
- `backend/tests/test_answer_integrity.py`
- ADR 0001 (refusal is not a repairable failure)
