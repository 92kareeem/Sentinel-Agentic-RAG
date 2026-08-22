# Phase 2 Implementation Plan

## Goal
Turn Sentinel from a strong MVP into a more product-like AI assistant with better context handling, stronger evaluation, and a more polished user experience.

## Priority 1 — Conversation memory
- Add conversation history to the query API contract.
- Pass previous turns to the synthesizer so follow-up questions are grounded in prior dialogue.
- Surface recent conversation context in the frontend.
- Add regression coverage for the new prompt behavior.

## Priority 2 — Evaluation and observability
- Add a richer eval harness with faithfulness, retrieval hit-rate, latency, and cost metrics.
- Persist eval results to a dashboard-friendly report.
- Track answer confidence and refusal quality over time.
- Implemented: lightweight offline faithfulness scoring and markdown report generation for repeatable eval runs.

## Priority 3 — UX upgrades
- Add streaming answer rendering.
- Show clearer citation highlighting and source snippets.
- Support chat history and follow-up refinement.

## Priority 4 — Production hardening
- Move ingestion to an async background pipeline.
- Add retry and timeout handling for external LLM calls.
- Introduce better caching and index refresh behavior.

## Priority 5 — Advanced AI features
- Support multi-modal documents.
- Add reranking and metadata-aware retrieval.
- Introduce document summarization and question generation workflows.
