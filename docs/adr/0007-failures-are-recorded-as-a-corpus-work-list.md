# ADR 0007 — Failures are recorded as a work list for the document owner

**Status:** Accepted · **Date:** 2026-09-19

## Context

Sentinel calls itself self-healing. What it actually had was *recovery*: a
failed critic score triggers a query rewrite or a model escalation, and the
request either succeeds or refuses. That is genuinely useful and it is also
entirely within one request. The next person to ask the same question starts
from scratch and fails the same way. Nothing in the system gets better.

The gap is not technical. It is about **who owns the failure**.

When Sentinel refuses, it has established a fact about the corpus — "nothing
here covers parental leave for contractors". Today that fact goes nowhere. The
person who asked rephrases, or gives up and asks a colleague. The document
owner, the one person who could write the missing paragraph, never hears about
it. The organisation keeps paying the cost of the gap indefinitely, and the gap
is invisible precisely because the system handled it politely.

That is also the difference between a chatbot and a documentation product. A
chatbot's value is the answers it gives. A documentation product's value
includes telling you what your documentation does not say.

## Decision

Persist every poorly-handled question as a **case**, classify it
deterministically, and expose the aggregate as a knowledge-gap report addressed
to the document owner.

### The diagnosis taxonomy splits by who can fix it

| Diagnosis | Meaning | Owner |
|---|---|---|
| `NO_EVIDENCE_FOUND` | retrieval returned nothing in scope | document owner |
| `EVIDENCE_OFF_TOPIC` | passages found, none addressed the question | document owner |
| `ANSWER_UNVERIFIABLE` | evidence was there, verification failed | us |
| `CITATION_MISATTRIBUTED` | answered, but attribution needed repair | us |
| `CAPACITY_EXCEEDED` | a budget or deadline fired first | neither |

This split is the point of the whole feature. "8% failure rate" tells a
business nothing about whether to write a policy or file a bug. Only the first
two are reported as gaps: telling an owner to write a document that would not
have helped is worse than telling them nothing, and a report that cries wolf
gets read once.

`CITATION_MISATTRIBUTED` is recorded even though the answer **shipped**. It is
a success with a quality signal attached, and it is the only way anyone learns
the corpus contains near-duplicate or superseded passages.

### Diagnosis is deterministic

Derived from graph state — retrieved chunk count, refusal reason, repaired
citation count. No LLM call.

An LLM classifier would be more nuanced. It would also mean every failure costs
another provider call at exactly the moment the system is already struggling,
and would put diagnosis on the same rate limit as answering. A burst of hard
questions must not make the thing that *explains* the burst the next thing to
fail. (This is not hypothetical: a Groq rate-limit storm has already turned CI
red on this project once.)

### Questions are clustered before reporting

The unit of work for a document owner is a topic, not a question. Six people
asking the same thing six ways is one missing paragraph; a flat list of 200
refusals is a report nobody reads twice.

Clustering is greedy single-pass on shared content words, with overlap measured
against the **smaller** question rather than the union — "refund window?" and
"what is the refund window for annual plans bought in the EU?" are the same
gap, and Jaccard would score them far apart purely because one is longer.

Term overlap rather than embeddings: free, stateless, cannot fail on a provider
outage, and **explainable** to the person acting on it, which matters more in a
report than marginal accuracy. Embedding the questions is the natural upgrade
if clusters get noisy at real volume; the interface does not change.

### Storage adds no AWS resources

Cases live in the **existing** documents table under `pk="CASE#<owner>"` with a
time-ordered sort key, so a 30-day window is one Query with a key condition and
no filter scan. Same pattern as the publication lock row (`LOCK#index`).

A dedicated table would be cleaner in isolation and is the right call at
volume. Here it buys nothing and costs a resource against a hard zero-cost
constraint.

### Volume is counted separately from cases

The casebook holds failures only — recording every success would turn it into a
query log and the gap report into something nobody scans.

But "answer rate" is the one number a manager acts on, and a rate needs a
denominator that failures alone cannot provide. So volume is a per-owner,
per-day pair of integers incremented with DynamoDB `ADD`: atomic, needs no
prior read, and two instances answering at once cannot lose a count.

Deriving the rate from the casebook instead would have reported a number near
zero and made a healthy system look broken. A fresh workspace reports 100%, not
0% — an empty denominator is not a bad score.

### Capture can never fail a request

Persistence is wrapped at the call site rather than per-writer, so the
guarantee holds however many things it grows to record.

A case is a byproduct of answering. Letting its write fail the request would
mean a user who asked a hard question loses their answer *because* it was hard
— inverting the purpose of a feature whose entire job is to make hard questions
better over time.

This also fixed a pre-existing defect on the same path: `put_trace` ran
unguarded, so a DynamoDB blip turned a finished answer into a 500.

## Alternatives considered

**Thumbs up/down feedback instead.** Not rejected — deferred, and it is a
different signal. User feedback is sparse, biased toward annoyance, and needs
review before it can be trusted. A refusal is an unambiguous, unsolicited,
system-generated fact. Start with the signal that needs no one to volunteer it.

**Materialise the report on a schedule.** Rejected under the zero-cost
constraint: a scheduled job needs something to run it, which means an always-on
process or an EventBridge rule firing whether or not anyone will read the
result. Computing on read costs nothing when nobody asks, and the clustering is
milliseconds over a single-partition query at this volume.

**Store the rejected draft in the case.** Rejected for the record; kept on the
state instead. The question is the durable artifact — it becomes a gap line and
later a regression test — while a rejected draft is a transient symptom of one
model run, and persisting unverified model text invites it to be shown.

## Consequences

- Refusals stop being dead ends and become the input to the next improvement.
- The report is the foundation for the next phase: promoting recurring cases
  into the golden dataset so a fixed gap stays fixed and the system cannot
  regress into the same failure twice.
- Storage grows with failures rather than with traffic, which is the right
  shape: a system that is working well writes almost nothing.

## Related

- `backend/app/learning/casebook.py`
- `backend/app/api/routes_insights.py`
- `backend/tests/test_casebook.py`, `backend/tests/test_insights_api.py`
- `frontend/src/components/InsightsModal.tsx`
