# ADR 0001 — A refusal is an outcome, not a failure to repair

**Status:** Accepted · **Date:** 2026-09-05

## Context

Sentinel's graph runs `synthesizer → critic → grounding_check`, with a bounded
repair loop (`repair_rewrite`, then `repair_escalate`) attached to the failure
edges of both gates.

The synthesizer is instructed to emit the exact token `INSUFFICIENT_CONTEXT`
when the retrieved evidence does not support an answer. That is a *correct*
outcome — refusing beats guessing, and it is the behaviour the product is
sold on.

The critic, however, is an LLM-as-judge scoring two axes, one of which is
`relevance = does the answer address the question?`. A refusal addresses
nothing, so it scores near zero on relevance. The routing table then treated
that low score exactly like a bad answer and sent it into repair.

Measured on the 20-question golden set (`evals/report.md`, 2026-08-26), the
three unanswerable questions each ran two repair rounds:

| Question | Latency | Repairs |
|---|---|---|
| 18 — parental leave policy | 24,141 ms | 2 |
| 19 — cloud provider | 24,577 ms | 2 |
| 20 — last quarter revenue | 25,341 ms | 2 |

Against a p50 of 9,548 ms, this single path *was* the system's p95 (25,341 ms).
The second of those rounds escalates to the 120b model, so it is also the most
expensive path in the system — spent entirely to re-derive a refusal that the
first round had already produced correctly.

## Decision

Route on the synthesizer's output **before** the critic sees it
(`_route_after_synthesizer` in `agents/graph.py`):

- answer is not the sentinel → `critic` (unchanged behaviour)
- sentinel, `attempt == 0`, budget allows → `repair_rewrite`
- sentinel, otherwise → `refusal`

Additionally, centralize the sentinel as `is_insufficient_context()` in
`models/schemas.py`, tolerant of decoration (trailing period, quotes, case,
an appended citation tag).

## Alternatives considered

**1. Refuse immediately on the first sentinel, with no repair at all.**
Rejected. `repair_rewrite` reformulates the query and *re-runs retrieval* —
"the evidence isn't here" genuinely can mean "retrieval missed it", and a
second query formulation can recover a real answer. Removing that round would
have traded latency for false refusals, and the eval currently records 0/17
false refusals — a number worth protecting.

**2. Teach the critic that a warranted refusal scores relevance = 1.0.**
Rejected. It costs an LLM call to ask a judge about an answer we have already
identified deterministically, it makes correctness depend on the judge
following an instruction it can ignore, and it leaves the escalation path
reachable for refusals.

**3. Let escalation run once for refusals.**
Rejected on principle, not just cost. `repair_escalate` only swaps in the
bigger model against the **same evidence set** — no new information enters the
system. If the small model reports that the evidence lacks the answer, a larger
model reading identical text will nearly always agree (~15s for nothing), and
in the case where it *disagrees*, it has talked itself into a claim the
evidence does not support. That is precisely the failure this system exists to
prevent, so the rare "win" is actually the bad outcome.

## Consequences

Measured on the same corpus (index v3), same judge, same dataset:

| Metric | Before | After |
|---|---|---|
| Latency p95 | 25,341 ms | 12,057 ms (**−52%**) |
| Refusal latency | 24.1 / 24.6 / 25.3 s | 12.1 / 11.2 / 10.5 s |
| Mean tokens/query | 1,950 | 1,609 (**−17%**) |
| Mean faithfulness | 0.950 | 0.955 |
| Unanswerable refusal-rate | 100% | 100% |
| False refusals | 0/17 | 0/17 |
| Retrieval hit-rate | 100% | 100% |

Refusals still cost ~11s because one rewrite round is deliberately retained.
Driving that lower means giving up the retrieval second chance, which is a
correctness trade we are explicitly not making.

**Tested by:** `backend/tests/test_graph_routing.py` — in particular
`test_refusal_after_a_rewrite_goes_straight_to_refusal_never_escalates`, which
fails if escalation-on-refusal is ever reintroduced.
