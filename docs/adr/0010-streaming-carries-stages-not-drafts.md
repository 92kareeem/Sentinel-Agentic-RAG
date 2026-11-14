# ADR 0010 — Streaming carries stages, never drafts

**Status:** Accepted · **Date:** 2026-11-14

## Context

A verified answer takes seconds: retrieval, drafting, an LLM judge, a
deterministic grounding gate, and sometimes a repair round. Users read a
silent spinner as "broken". The standard fix is token streaming — show the
answer as the model writes it.

For Sentinel that fix is the bug ADR 0005 exists to prevent. The text a model
writes is a *draft* until the critic and the grounding gate have passed it.
Streaming it means showing readers, live, exactly the text the system may be
about to refuse — and a refused draft, once read, has done its damage: a
reader takes away the claim, not the retraction that follows it.

## Decision

`POST /v1/query/stream` sends Server-Sent Events of three kinds:

- `stage` — one per completed graph node: its name, a human label, the
  attempt number, and the step's timing, tokens and an **allow-listed**
  subset of its metadata (model, chunk counts, rows excluded by access,
  grounding pass/fail and strip ratio, refusal reason).
- `result` — exactly the body `POST /v1/query` returns, sent last.
- `error` — a provider failure after the stream opened, as `{status, detail}`.

Metadata is an allow-list, not a deny-list. The critic's `why` and the
rewritten search query are model output that can quote or paraphrase an
unverified draft, so they are withheld from the stream (they stay in the
stored trace, which an owner reads after the fact).

Guardrails run **before** the stream opens, so a blocked request still gets
its real status code (401, 413, 422, 429, 404, 409) rather than a 200 whose
first event is an error.

Both routes share one pipeline (`api/query_pipeline.py`), so the refusal
contract, the cache and the bookkeeping cannot differ between them.

## Consequences

- The reader sees progress — "Verifying grounding", "Repairing: rewriting the
  search" — which also teaches them what the product does.
- Time to first answer token is unchanged. That is the accepted cost.
- Behind a buffering proxy (AWS Lambda Function URLs via Mangum buffer the
  response) every event arrives at once at the end. The result is identical;
  only the progress display is lost. A Lambda response-streaming adapter is
  the upgrade path and changes nothing in this contract.

**Tested by:** `backend/tests/test_streaming_and_cache.py` — in particular
`test_a_rejected_draft_never_reaches_the_stream_at_all`, which runs the real
graph against a fabricated figure and asserts it appears in no event.
