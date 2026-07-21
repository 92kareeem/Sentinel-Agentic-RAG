# Sentinel — Document Self-Healing Agentic RAG

A guardrailed, self-healing retrieval-augmented generation platform. Built end-to-end and deployed on AWS free tier.

**Stack:** FastAPI · LangGraph · Hybrid FAISS + BM25 (RRF) · Groq (Llama 3.1 8B routing/critic, 70B escalation) · AWS Lambda + API Gateway + DynamoDB + S3 + CloudFront · pytest · GitHub Actions

---

## What it does

Most RAG systems are one-shot: retrieve, generate, ship. When retrieval misses or the model hallucinates, the user sees the failure.

Sentinel wraps the pipeline in a LangGraph agent that grades its own output and repairs it before responding.

```
                ┌─────────┐
   query ──────▶│ router  │─── simple? ──▶ direct answer
                └────┬────┘
                     │ complex
                     ▼
                ┌─────────┐    ┌──────────────┐
                │retriever│───▶│ synthesiser  │
                └─────────┘    └──────┬───────┘
                                      ▼
                                ┌─────────┐
                                │ critic  │
                                └────┬────┘
                                     │ low-confidence
                                     ▼
                                ┌─────────┐
                                │ repair  │──── loop back to retriever
                                └─────────┘
```

- **Router** — Groq 8B classifies query complexity and routes cheaply.
- **Hybrid retriever** — FAISS (dense, IndexFlatIP) + BM25 (sparse) fused with Reciprocal Rank Fusion (k=60). Sparse recall for exact terms, dense recall for meaning.
- **Synthesiser** — Groq 70B, grounded on retrieved chunks with data-grounded prompting.
- **Critic** — evaluates the answer against retrieved context. Faithfulness and relevance scored.
- **Repair loop** — on low confidence, rewrites the query and re-retrieves. Bounded to prevent runaway loops.
- **Guardrails** — input and output. Blocks prompt-injection patterns, PII leakage, off-topic drift.
- **Full request tracing** — every node emits structured logs; traces stored in DynamoDB for replay and debugging.

The point isn't the framework choices. The point is the platform grades itself, catches its own failures, and only ships answers it can defend.

---

## Status

- [x] **P1** — Ingestion pipeline and hybrid index (`make ingest`)
- [x] **P2** — LangGraph agent: router → retriever → synthesiser → critic → repair, with Groq small/large model routing
- [x] **P3** — FastAPI service, input/output guardrails, pytest suite
- [x] **P4** — AWS deploy: Lambda (container image), API Gateway, DynamoDB (traces + API keys), S3 (index artifacts), CloudFront (frontend distribution)
- [x] **P5** — Evaluation harness with faithfulness / answer-relevance / retrieval-hit metrics, GitHub Actions CI
- [x] **P6** — TypeScript frontend

---

## Architecture decisions

| Decision                     | Choice                                                | Why                                                                    |
| ---------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------- |
| Orchestration                | LangGraph                                             | Explicit state machine; conditional edges make the repair loop trivial |
| Vector store                 | FAISS `IndexFlatIP` in Lambda memory, artifacts on S3 | Exact search, zero servers, free tier                                  |
| Sparse retrieval             | BM25                                                  | Exact-term recall the dense index misses                               |
| Fusion                       | Reciprocal Rank Fusion (k=60)                         | No score calibration needed across dense/sparse                        |
| LLM provider                 | Groq                                                  | Fast inference, generous free tier                                     |
| Small/large split            | Llama 3.1 8B (router, critic) + 70B (synthesis)       | 80% of cost lives on the small model; escalate only when needed        |
| Serving                      | AWS Lambda container image behind API Gateway         | Cold start acceptable for demo; scales to zero; free tier              |
| State                        | DynamoDB (traces, API keys)                           | Serverless, single-digit-ms reads, no schema migrations                |
| Auth                         | API Gateway usage plans + hashed keys in DynamoDB     | Two layers of protection, no Cognito overhead                          |
| Frontend                     | React + TypeScript on CloudFront                      | Static hosting, cheap, edge-cached                                     |
| Tests                        | pytest, GitHub Actions on push                        | Ingestion, retrieval, agent nodes, guardrails, end-to-end              |

---

## Repo layout

```
backend/            FastAPI service, LangGraph nodes, guardrails, retrieval
frontend/           React + TypeScript chat UI
infra/              IaC: Lambda, API Gateway, DynamoDB, S3, CloudFront
evals/              Evaluation harness, golden dataset, report generator
docker/             Lambda container image
docs/               Sample documents for the demo corpus
.github/workflows/  CI — lint, test, eval on push
```

---

## Quickstart (local)

```bash
# Environment: venv must live outside cloud-synced folders (OneDrive corrupts native DLLs)
# and be built from a standalone CPython (conda-derived venvs break torch DLL init).
uv venv C:/venvs/sentinel --python 3.12
uv pip install -e ".[dev]" --python C:/venvs/sentinel/Scripts/python.exe

cp .env.example .env                   # fill GROQ_API_KEY
make ingest                            # build index/ from ./docs, runs smoke test
make test                              # full pytest suite
make eval                              # run evaluation harness → evals/report.md
make serve                             # local FastAPI on :8000
```

---

## Deploy

```bash
make build-image                       # build Lambda container
make deploy                            # infra stack up: Lambda, API GW, DynamoDB, S3, CloudFront
```

Deployment is provisioned via IaC in `infra/`. Endpoint URL is issued at deploy time; auth is via hashed API key.

---

## Why this project

Every AI Engineer job ad in 2026 mentions RAG, agents, evals and guardrails as bullet points on a wishlist. This is what those bullet points actually look like when they meet each other in production: a system that routes cheaply, retrieves hybrid, grades itself, repairs when it fails, and refuses to answer when it can't be sure. Every architecture decision above is a real trade-off I made — and can defend in an interview.

Built by [Syed Abdul Kareem Ahmed](https://linkedin.com/in/92kareem). Relocating to Melbourne, immediate start on offer.

## License

MIT
