# Sentinel — Document Self-Healing Agentic RAG

A guardrailed, self-healing retrieval-augmented generation platform for document Q&A: every answer is cited to a page, and unverifiable answers are refused rather than guessed.

> **Verification status.** Local and CI paths are verified by the test suite and the eval
> harness. The AWS deployment in `infra/` has been executed end-to-end and is serving:
> CloudFront frontend, Lambda container behind a Function URL, DynamoDB, S3, and the Groq
> key in an SSM SecureString. See [Known limitations](#known-limitations) for what is
> still not true of it.

**Stack:** FastAPI · LangGraph · Hybrid FAISS + BM25 (RRF) · Groq (`openai/gpt-oss-20b` synthesis/critic, `120b` escalation) · AWS Lambda + DynamoDB + S3 + CloudFront · pytest · GitHub Actions

---

## What it does

Most RAG systems are one-shot: retrieve, generate, ship. When retrieval misses or the model hallucinates, the user sees the failure.

Sentinel wraps the pipeline in a LangGraph agent that grades its own output and repairs it before responding.

```
              ┌────────┐    ┌───────────┐    ┌─────────────┐
  query ─────▶│ router │───▶│ retriever │───▶│ synthesiser │
              └────────┘    └───────────┘    └──────┬──────┘
               picks the      always runs           │
               model, not     — there is no         ▼
               whether to     answer-without-  ┌────────┐
               retrieve       retrieval path   │ critic │
                                               └───┬────┘
                          ┌────────────────────────┤
                          │ low confidence         │ passes
                          ▼                        ▼
                    ┌──────────┐            ┌───────────┐
                    │  repair  │            │ grounding │
                    │ rewrite  │            │   gate    │
                    │ escalate │            └─────┬─────┘
                    └────┬─────┘      deterministic│
                         │ re-runs                 │
                         │ retrieval    ┌──────────┴──────────┐
                         └──────────────┤ ships / repairs /   │
                                        │ refuses             │
                                        └─────────────────────┘
```

Every substantive sentence carries a citation, and the grounding gate checks it:
the numbers in a claim must appear in the passage it cites, and a citation that
names the wrong passage is re-pointed at the one that actually supports the
claim rather than accepted. What cannot be supported is stripped; if most of an
answer is stripped, the whole answer is refused rather than shipped thin.

- **Router** — a keyword heuristic selects the small or large model. It used to spend an
  LLM call on this; measurement showed that call had never actually worked, and that
  repairing it made quality *worse*. See [ADR 0002](docs/adr/0002-router-uses-a-heuristic-not-an-llm.md).
- **Hybrid retriever** — FAISS (dense, IndexFlatIP) + BM25 (sparse) fused with Reciprocal Rank Fusion (k=60). Sparse recall for exact terms, dense recall for meaning.
- **Synthesiser** — grounded strictly on retrieved chunks; document text is fenced and labelled untrusted so content inside an uploaded file cannot act as an instruction.
- **Critic** — evaluates the answer against retrieved context. Faithfulness and relevance scored.
- **Repair loop** — on low confidence, rewrites the query and re-retrieves. Bounded to prevent runaway loops.
- **Guardrails** — input and output. Blocks prompt-injection patterns, PII leakage, off-topic drift.
- **Grounding gate** — deterministic, after the critic. Cannot hallucinate, because it contains no model: it checks numbers and vocabulary against the cited passage, repairs misattributed citations, and strips what nothing supports.
- **Knowledge-gap report** — every question the documents could not answer is recorded, diagnosed and clustered into a ranked work list for whoever owns the documents. A refusal is a fact about the corpus, and it should reach the person who can fix it.
- **Full request tracing** — every node emits structured logs; traces stored in DynamoDB for replay and debugging.

The point isn't the framework choices. The point is the platform grades itself, catches its own failures, and only ships answers it can defend.

---

## Status

- [x] Ingestion pipeline and hybrid index (`make ingest`)
- [x] LangGraph agent: router → retriever → synthesiser → critic → grounding → repair → refusal
- [x] FastAPI service, guardrail chain, 250+ backend tests (unit + HTTP contract + adversarial + concurrency + end-to-end)
- [x] Document registry with explicit lifecycle, ownership and tenant isolation
- [x] Page-aware PDF pipeline with typed failure modes (encrypted / scanned / corrupt)
- [x] Atomic, versioned index publication, serialized across processes by a DynamoDB lease lock
- [x] AWS deploy: Lambda container, DynamoDB, S3, CloudFront, SSM SecureString — **executed and serving**
- [x] Evaluation harness (faithfulness / completeness / retrieval-hit / refusal-rate), GitHub Actions CI
- [x] TypeScript frontend, light/dark, markdown answers, click-through citations
- [x] Knowledge-gap report: failures recorded, diagnosed, clustered, and surfaced to document owners
- [x] Reader feedback: answers can be reported wrong or incomplete, with a correction; refusals can be reported as false
- [x] Regression ratchet: a reviewed failure becomes a per-question test that, once it passes, can never fail unnoticed (ADR 0008)

### Known limitations

These are deliberate scope boundaries, not oversights:

- **No OCR.** Image-only/scanned PDFs are detected and rejected with
  `UNSUPPORTED_SCANNED_DOCUMENT` rather than indexed as empty.
- **Answers are not token-streamed.** `/v1/query` returns one JSON body after the
  graph completes; the UI renders it progressively (labelled as such, not as streaming).
- **Ingestion is synchronous.** Fine for the 1 MB / 200-page limit this targets. A
  durable queue (S3 event → SQS → worker) is the right shape beyond that; the
  previous in-process daemon thread was removed because Lambda freezes on return.
- **Follow-up questions are not resolved before retrieval.** "What about contractors?"
  is embedded literally. The synthesiser sees the conversation and often recovers, but
  retrieval searched for the wrong thing, so on a follow-up the evidence may simply not
  be there.
- **Whole-document questions are served by top-k similarity.** "List every exception"
  needs coverage; top-k returns the *most similar* k passages, which is a different
  thing. The answer looks complete either way, and nothing detects the difference yet.
- **Supersession is not modelled.** Two documents stating different refund windows are
  two pieces of evidence. Nothing marks one as current, and upload order is not
  authority.
- **One identity per API key.** There is a tenant, but no concept of a *person*, so
  there are no roles, no per-user document permissions and no audit of who asked what.
- **Regressions run against the eval corpus only.** A promoted case is tested in CI
  against `./corpus`. A tenant's own documents would need a per-tenant regression run
  against that tenant's index. Not built yet.
- **The browser holds an API key.** `VITE_API_KEY` is baked into the bundle and is
  extractable by anyone who loads the page. Acceptable for local development and a
  quota-limited demo; not acceptable as production authentication.

---

## Architecture decisions

| Decision                     | Choice                                                | Why                                                                    |
| ---------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------- |
| Orchestration                | LangGraph                                             | Explicit state machine; conditional edges make the repair loop trivial |
| Vector store                 | FAISS `IndexFlatIP`, immutable versioned artifacts on S3 | Exact search, zero servers; versioning makes publication atomic     |
| Document identity            | Server-generated uuid + DynamoDB registry             | Filenames are neither unique across users nor stable across re-uploads |
| Sparse retrieval             | BM25                                                  | Exact-term recall the dense index misses                               |
| Fusion                       | Reciprocal Rank Fusion (k=60)                         | No score calibration needed across dense/sparse                        |
| LLM provider                 | Groq                                                  | Fast inference, generous free tier                                     |
| Small/large split            | `gpt-oss-20b` (synthesis, critic) + `120b` (escalation) | Most cost lives on the small model; escalate only when repair needs it |
| Query routing                | Keyword heuristic, no LLM call                        | Measured: the LLM classifier added latency and tokens, and cost a false refusal (ADR 0002) |
| Refusal handling             | Short-circuited before the critic                     | Measured: repairing refusals *was* the p95 — 25.3s → 12.1s (ADR 0001) |
| Refusal text                 | Generated from a reason code, never from model output | The rejected draft is exactly what we decided not to stand behind (ADR 0005) |
| Concurrent publication       | DynamoDB lease lock in the existing documents table    | Conditional write is atomic at the database; no new resource, no IAM change |
| Learning from failures       | Deterministic diagnosis, clustered on term overlap     | Costs no provider call, so a burst of hard questions cannot break diagnosis too (ADR 0007) |
| Learning from readers        | Feedback on a trace; one verdict per reader, atomically | The failure no check can see is a confident wrong answer — only the reader can report it (ADR 0008) |
| Not repeating mistakes       | Per-question regression suite, human-promoted, pending → enforced | An average gate lets one question fall from 1.0 to 0.0 and still pass (ADR 0008) |
| Serving                      | AWS Lambda container image behind API Gateway         | Cold start acceptable for demo; scales to zero; free tier              |
| State                        | DynamoDB (traces, API keys)                           | Serverless, single-digit-ms reads, no schema migrations                |
| Auth                         | API Gateway usage plans + hashed keys in DynamoDB     | Two layers of protection, no Cognito overhead                          |
| Frontend                     | React + TypeScript on CloudFront                      | Static hosting, cheap, edge-cached                                     |
| Tests                        | pytest on every PR; evals nightly or on a `run-evals` label | Unit CI is fast and always on; the eval suite paces itself around a rate limit, so it is opt-in |

---

## Repo layout

```
backend/            FastAPI service, LangGraph nodes, guardrails, retrieval, casebook
frontend/           React + TypeScript chat UI
infra/              IaC: Lambda, Function URL, DynamoDB, S3, CloudFront
evals/              Evaluation harness, golden dataset, report generator
docker/             Lambda container image
corpus/             The demo business documents — the ONLY thing that gets indexed
docs/               Engineering documentation and ADRs (deliberately NOT corpus)
.github/workflows/  ci.yml (every PR) and evals.yml (nightly, or per-PR via label)
```

---

## Quickstart (local)

```bash
# Environment: venv must live outside cloud-synced folders (OneDrive corrupts native DLLs)
# and be built from a standalone CPython (conda-derived venvs break torch DLL init).
uv venv C:/venvs/sentinel --python 3.12
uv pip install -e ".[dev]" --python C:/venvs/sentinel/Scripts/python.exe

cp .env.example .env                   # fill GROQ_API_KEY
make ingest                            # build index/ from ./corpus, runs smoke test
make test                              # full pytest suite
make eval                              # run evaluation harness → evals/report.md
make serve                             # local FastAPI on :8000

# The learning loop: review what readers reported, turn it into regression tests
python evals/promote.py list                        # failures not yet in the suite
python evals/promote.py promote <case_id>           # reader's correction becomes the reference
python evals/run.py --regressions-only              # check just the ratchet after a fix
python evals/run.py --lock                          # lock pending regressions that now pass
```

---

## Deploy

```bash
export AWS_ACCOUNT_ID=... GROQ_API_KEY=...
make deploy            # == bash infra/deploy.sh
```

`infra/deploy.sh` is idempotent and creates/updates: S3 buckets (+ lifecycle, CORS),
the four DynamoDB tables, the least-privilege IAM role, the ECR repo and image, the
Lambda function and its URL, and finally syncs the local index — publishing the
version directory *before* the pointer so a cold-starting Lambda never reads a torn
index. It ends with a `/healthz` smoke test.

The container image fetches the quantized MiniLM at build time rather than copying a
gitignored `models/onnx/`, so a **fresh clone builds** (this is enforced by a CI job).

Auth is a hashed API key in DynamoDB, checked per request. The frontend ships only a
non-privileged, quota-limited demo key; the admin key must never be built into it.

---

## Why this project

Every AI Engineer job ad in 2026 mentions RAG, agents, evals and guardrails as bullet points on a wishlist. This is what those bullet points actually look like when they meet each other in production: a system that routes cheaply, retrieves hybrid, grades itself, repairs when it fails, and refuses to answer when it can't be sure. Every architecture decision above is a real trade-off I made.

Built by [Syed Abdul Kareem Ahmed](https://linkedin.com/in/92kareem). 
