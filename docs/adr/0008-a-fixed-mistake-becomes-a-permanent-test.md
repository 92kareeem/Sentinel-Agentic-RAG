# ADR 0008 — A fixed mistake becomes a permanent test

**Status:** Accepted · **Date:** 2026-09-26

## Context

ADR 0007 gave failures somewhere to go: a casebook, and a gap report addressed
to the person who owns the documents. That closed half the loop. The other half
was missing in two places.

**The system could only learn from failures it noticed itself.** Every case came
from the graph's own state: a refusal, a rejected draft, a repaired citation.
One kind of failure could never show up at all: an answer that passed every
gate, shipped with citations, and was wrong. If any upstream check could detect
it, that check would already have refused. Only the reader can say so, and it
is the failure a reader remembers.

**Nothing stopped a fixed mistake from coming back.** The eval gate is an
average — mean faithfulness ≥ 0.75 across twenty questions. One question can
fall from 1.0 to 0.0, move the mean by 0.05, and pass. An average is the right
gate for "is the system good overall?" and structurally incapable of answering
"did this specific mistake return?"

## Decision

### 1. Readers can report an answer as wrong

`POST /v1/feedback` takes a trace id, one of `HELPFUL` / `INCORRECT` /
`INCOMPLETE`, and an optional correction. Negative verdicts open a case with a
new diagnosis (`USER_REPORTED_INCORRECT` / `_INCOMPLETE`), shown on the gap
report beside the refusals. Refusals can be rated too: "you refused, but the
handbook covers this" is a false refusal, reported by the one person positioned
to notice it.

The correction is the point. A thumbs-down says a question failed; a correction
says what passing looks like. Only the second can become a test.

Guarantees, each tested and each checked by reintroducing the bug it prevents:

- **One reader, one answer, one verdict.** The claim on (owner, trace), the
  case, and the counters are written as a single DynamoDB transaction. As three
  sequential writes, a partial failure could leave a reader told "already rated"
  for a report that was never stored. And deduplicating on the case alone would
  let "helpful" then "wrong" count twice, because HELPFUL opens no case.
- **The question comes from the trace, never from the client.** Corrections
  feed the test suite; a client that supplied the question could attach it to a
  real answer and plant a "known correct" answer.
- **Another tenant's trace is 404, not 403.** Confirming a trace exists would
  already leak something.

### 2. A separate, per-question regression suite

`evals/regressions.json`, gated **per question**, runs after the golden set. It
is a separate file because mixing its rows into the golden set would quietly
change what the golden set's historical averages mean.

- Items start **pending**: tracked, reported, not blocking. A failure is
  promoted when someone confirms it, usually before it is fixed. Enforcing it
  at that point would turn CI red at the one moment nobody can fix it.
- Once a pending item passes, it is **locked** and becomes **enforced**
  (`run.py --lock` or `promote.py lock`). From then on, if it fails, the build
  fails.
- An enforced failure is **re-run once** before it can fail the build. A single
  judge call is noisy, and a ratchet that goes red on noise gets switched off,
  which undoes the whole mechanism.
- An **unmeasured** item (rate limit, outage) is never a failure. That was the
  lesson of the eval work in September, where three separate bugs reported
  provider limits as quality regressions.

### 3. Promotion is a human act

`evals/promote.py` lists unpromoted cases and promotes one on request. It is
never automatic. A correction is a claim by whoever typed it; auto-promoting
corrections would let any user write the test suite. This was demonstrated
during end-to-end testing: a reader's wrong correction ("the US restocking fee
is 15%"), promoted without review and locked, failed CI against a system that
correctly said 5%. The reviewer's `--reference` always overrides the reader's
correction.

### 4. "Must refuse" is never inferred

The obvious design — promote every refusal as "must refuse forever" — puts the
ratchet at war with the gap report. The report tells the owner to add the
missing document. The moment they do, the right behaviour flips to answering,
and a must-refuse test would fail CI for the exact fix the product asked for.

So a refusal is promoted as `answer` once the gap is closed, which locks the fix
in. `refuse` is reserved for questions someone has **decided** are permanently
out of scope, and it requires a written `--note`.

### 5. Nothing is deleted

A lesson that stops being true (the refund window changed) is **retired** with
a reason. Two guards make a quiet deletion visible:

- ids are dense across active and retired (deleting R002 leaves a gap), and
- the suite records how many ids it has ever **issued**, because dense
  numbering alone cannot see the deletion of the *newest* item. That blind spot
  was found during end-to-end testing, and it matters most: the newest lesson
  is the one still failing and the one most tempting to delete.

The suite is validated before any quota is spent, by `save()`, and by a unit
test against the committed file.

## The loop, end to end

```
refusal ─► gap report ─► owner adds the document ─► promote(answer) ─► lock
                                                    can never go back to refusing

wrong answer ─► reader's correction ─► human review ─► promote(answer) ─► lock
                                                    can never be wrong that way again
```

Verified live against the real server, retrieval and Groq on 2026-09-26: ask →
feedback → Coverage report → `promote.py list/promote` → `run.py` (pending
failure does not block, exit 0) → lock → enforced failure confirmed on re-run
(exit 1) → retire → exit 0. Tampering (deleting a middle item, deleting the
newest) was rejected before any LLM call.

## Consequences

- **Every enforced regression costs quota on every run**, about one golden
  question's worth each. This is the price of permanence. Refuse-type items
  need no judge and are cheap. Moving evals to nightly runs (the evals.yml change of 2026-09-20) keeps
  it off the PR path.
- **Regressions run against the eval corpus.** A case about a customer's own
  handbook can only be promoted into CI if CI indexes that handbook. In this
  repository that holds, because CI indexes `./corpus`. A multi-tenant
  deployment would need per-tenant regression runs against each tenant's own
  index. That is the natural next step, and it is deliberately not built yet.
- **The guards stop an accidental or casual deletion, not a determined one.**
  Someone who edits `issued` and renumbers can hide a deletion — but only
  through a deliberate multi-line change in a reviewed diff. The complete fix
  is a CI check that every id present on `main` still exists on the PR branch.
  That is noted as future work rather than claimed.
