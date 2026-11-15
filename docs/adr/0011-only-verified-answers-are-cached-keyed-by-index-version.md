# ADR 0011 — Only verified answers are cached, keyed by index version and visibility

**Status:** Accepted · **Date:** 2026-11-14

## Context

The same questions recur — "what is the refund window?" is asked every day.
Each one costs a full graph run: retrieval, a synthesis call, a judge call,
on a free tier rate-limited per minute. Caching is the obvious lever, and
caching RAG answers is the obvious way to serve a stale or leaked answer.

Two failure shapes matter:

1. **Stale.** A document is updated; the cache keeps serving the old answer
   with citations that no longer exist.
2. **Leaked.** An answer built from a document one user may read is served to
   a user who may not.

TTL-based invalidation addresses neither: it is a guess about how long the
corpus stays the same, and it knows nothing about who is asking.

## Decision

Cache in-process (`rag/answer_cache.py`), with a key built from everything an
answer depends on:

| Component | Why |
|---|---|
| index version | any publish changes it, so every corpus change invalidates every entry by construction |
| tenant, user, roles, clearance | visibility differs per caller |
| document scope | a scoped question is a different question |
| provider + model | a model switch is a deliberate quality change |
| question | normalised for case and whitespace only — never semantic |

Store only answers that went through the critic **and** the grounding gate.
Never store refusals (a budget refusal is transient, and a knowledge gap must
be re-tested once the missing document arrives), follow-up questions (they
mean something different in another conversation), or ungraded output.

A hit is re-issued under the new request's trace id, with a `cache` step, so
feedback, telemetry and the trace inspector work on it unchanged.

## Alternatives considered

**Semantic cache (embedding similarity).** Rejected. "Refund window for EU
customers" and "refund window for US customers" embed close together and
have different answers. A semantic cache serves one question's verified
answer to another question — confidently, with citations — which is the
answer-to-a-different-question failure ADR 0006 removed from the graph.

**Shared cache (ElastiCache / DynamoDB).** Not free tier, or adds a write per
answer. Per-container caching on Lambda has a lower hit rate, and the
telemetry tab reports the real rate rather than a projected one.

## Consequences

- A repeated question costs zero tokens and milliseconds.
- `ANSWER_CACHE_TTL_SECONDS=0` disables it entirely.

**Tested by:** `backend/tests/test_streaming_and_cache.py`.
