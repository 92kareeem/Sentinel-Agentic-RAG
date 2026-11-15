# Learning Sentinel, one idea at a time

Ten checkpoints, in the order the code depends on them. Each names the files
to read, the idea to take away, and an exercise that proves you have it. The
exercises run offline: no provider key, no AWS.

Set up once:

```bash
make install          # venv + frontend deps
make ingest           # index ./corpus
make test             # the backend suite, fully offline
```

---

## 1. A request is a state machine, not a chain

**Read:** `backend/app/agents/graph.py`, `backend/app/agents/state.py`

Every routing decision (answer, rewrite, escalate, refuse) is a conditional
edge in one file. Nodes mutate one typed state; nothing routes from inside a
node.

**Exercise:** In `backend/tests/test_graph_routing.py`, find the test that
fails if a refusal is ever escalated to the large model. Explain in one
sentence why escalating a refusal is not just slow but *wrong* (ADR 0001).

## 2. The question is not the search query

**Read:** `state.py` (`query` vs `search_query`), `agents/repair.py`, ADR 0006

Repair rewrites what retrieval searches for and never what the user asked.
Otherwise every downstream gate agrees on an answer to a question nobody
asked.

**Exercise:** Change `repair_rewrite_node` to write `state["query"]`, run
`make test`, and read which test catches it.

## 3. A deterministic gate behind an LLM judge

**Read:** `packages/sentinel-verify/src/sentinel_verify/grounding.py` (the gate, a standalone package), `guardrails/grounding.py` (the platform's adapter), `agents/critic.py`, ADR 0003

The critic is a model and can be fooled in the same direction as the model
it grades. The grounding gate contains no model: every number in a claim must
appear in the passage it cites, and a misattributed citation is re-pointed or
stripped.

**Exercise:** Open the control plane's chaos demo and run *Hallucination
trap*. The critic is deliberately fooled into 0.95. Read the trace: which node
rejected it, and how many times?

## 4. A refusal carries no model text

**Read:** `models/schemas.py` (`safe_refusal_text`), `api/query_pipeline.py`
(`finalize`), ADR 0005

Refusal prose is generated from a reason code. The rejected draft stays in
state for triage and never reaches a response body or a stream.

**Exercise:** `grep -n "rejected_draft" -r backend/app` and confirm no
response model contains it.

## 5. One retry loop for every provider

**Read:** `llm/http_chat.py`, `llm/registry.py`, `llm/anthropic_client.py`

Timeouts, jittered backoff, Retry-After and the circuit breaker exist once.
The request deadline is enforced *inside* the call, because a 45-second
budget was once measured taking 285 seconds when only the gaps between nodes
were checked.

**Exercise:** Set `LLM_PROVIDER=local` with Ollama running, ask a question,
and compare the trace's `provider` and `est_cost_usd` with a Groq run.

## 6. Cost is a rule, not a model call

**Read:** `agents/budget.py` (`pick_model`), `llm/pricing.py`, ADR 0002

The router picks a model with a keyword heuristic, because an LLM router was
measured costing tokens and a false refusal for no gain. The same reasoning
applies to cost: when time or tokens are thin, the cheap model is used, by
rule.

**Exercise:** Run the chaos demo's *Budget exhaustion*. Find
`downgraded: True` on the synthesizer step.

## 7. Access is enforced before scoring

**Read:** `guardrails/access.py`, `agents/retriever.py`, `rag/index_store.search`,
ADR 0009

Filtering after ranking lets forbidden rows crowd out the caller's own
evidence. Sentinel hands FAISS the allowed rows as a selector, so excluded
rows are never scored.

**Exercise:** Upload a document as *Finance analyst* with classification
CONFIDENTIAL and role `finance`. Switch to *Demo employee* and ask about it.
Open the trace: the retriever step reports how many rows were hidden.

## 8. A citation you can verify

**Read:** `rag/chunking.provenance_hash`, `frontend/src/lib/provenance.ts`

Each chunk carries `sha256(file checksum | start | end | text)`. The source
drawer recomputes it in the browser with the recipe the API publishes.

**Exercise:** Compute the hash yourself for any citation with the recipe in
`GET /v1/documents/{id}/chunks/{chunk_id}`, and check it against the response.

## 9. Failures become a work list

**Read:** `learning/casebook.py`, `api/routes_insights.py`, ADR 0007

A refusal is a fact about the corpus. It is diagnosed deterministically,
clustered with similar questions and ranked for the person who owns the
documents.

**Exercise:** Ask "What is the parental leave policy for contractors?". Open
the control plane, *Corpus gaps*, and use *Upload the missing document &
re-verify* with a one-paragraph Markdown policy. Watch the question get
asked again and answered.

## 10. A fixed mistake becomes a permanent test

**Read:** `learning/ratchet.py`, `evals/promote.py`, `evals/run.py`, ADR 0008

Averages hide regressions: one question can fall from 1.0 to 0.0 and move a
20-question mean by 0.05. The ratchet gates each promoted question on its own.
Promotion is a human act, lands as pending, and locks only after it passes.

**Exercise:** In the control plane's *Quality ratchet* tab (as *Tenant
admin*), promote the case from checkpoint 9 with a reference answer. Then:

```bash
git diff evals/regressions.json
make regressions          # needs a provider key: judges the answer
```

---

## Where to go next

- `docs/SENTINEL_SYSTEM_GUIDE.md` — the long-form architecture guide
- `docs/adr/` — every decision, the measurement behind it, and what it rules out
