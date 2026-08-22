# Phase 2 Progress Log

## Overview
This document tracks the Phase 2 evolution of Sentinel from a working MVP into a more product-like AI assistant.

## Phase 1 baseline
Completed before Phase 2:
- document ingestion and hybrid retrieval
- guardrailed query flow
- LangGraph-based self-healing agent loop
- AWS deployment prototype with S3, Lambda, DynamoDB, and CloudFront
- frontend demo experience

## Phase 2 objectives
1. Improve conversational context handling
2. Add stronger evaluation and observability
3. Upgrade the user experience with clearer citations and history
4. Harden the system for production-style reliability
5. Prepare the project for advanced AI features

## Phase 2 implementation log

### Phase 2.1 — Conversational memory
Status: Completed

What changed:
- Added conversation history to the query API request schema.
- Passed prior turns into the synthesizer prompt so follow-up questions can reference earlier dialogue.
- Stored recent conversation turns in frontend state for a lightweight chat-like experience.
- Added a regression test to verify that previous turns influence the prompt passed to the LLM.

Files involved:
- backend/app/models/schemas.py
- backend/app/api/routes_query.py
- backend/app/agents/synthesizer.py
- frontend/src/api.ts
- frontend/src/App.tsx
- backend/tests/test_conversation_memory.py

### Phase 2.2 — Evaluation framework
Status: Completed

What changed:
- Added a lightweight offline evaluation helper for faithfulness scoring.
- Implemented report generation for mean faithfulness, hit-rate, refusal-rate, latency, and token usage.
- Added regression tests to verify the evaluator and markdown report output.
- Wired the evaluation utilities so they can be reused by the existing eval runner and future CI gates.

Files involved:
- backend/app/observability/evaluation.py
- backend/tests/test_evaluation.py

### Phase 2.3 — UX upgrades
Status: Completed

What changed:
- Added streaming-style answer rendering so the response appears progressively in the UI.
- Improved citation presentation with clearer chip-based navigation and a richer citation drawer.
- Kept the conversation context visible to make the experience feel more like a chat assistant.

Files involved:
- frontend/src/App.tsx
- frontend/src/components/AnswerPanel.tsx
- frontend/src/components/CitationDrawer.tsx
- frontend/src/styles.css

### Phase 2.4 — Production hardening
Status: Completed

What changed:
- Added an ingest job tracker so document indexing can be processed in a background-style flow rather than blocking the request path.
- Introduced a lightweight job state model so the system can report queued, running, completed, and failed ingestion progress.
- Added regression coverage for the new ingest job lifecycle.

Files involved:
- backend/app/rag/ingest_runtime.py
- backend/tests/test_ingest_hardening.py

### Phase 2.5 — Advanced AI features
Status: Planned

Planned work:
- Support multi-modal documents.
- Add reranking and metadata-aware retrieval.
- Introduce document summarization and task-oriented assistant flows.

## Verification notes
- Conversation memory regression test was executed successfully with:
  - C:/venvs/sentinel/Scripts/python.exe -m pytest backend/tests/test_conversation_memory.py
- Result: 1 passed
- Frontend build verified with:
  - npm run build
- Result: production build completed successfully
- Ingest hardening tests verified with:
  - C:/venvs/sentinel/Scripts/python.exe -m pytest backend/tests/test_ingest_hardening.py backend/tests/test_evaluation.py backend/tests/test_conversation_memory.py
- Result: 4 passed

## Next milestone
The next milestone is to implement the evaluation framework so the system can measure response quality in a repeatable way.
