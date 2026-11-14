# ADR 0009 — Access is enforced before scoring, not after ranking

**Status:** Accepted · **Date:** 2026-11-14

## Context

Sentinel had one identity per API key and one isolation rule: a chunk is
visible to the user who uploaded it. Enterprises need three more dimensions —
the organisation (tenant), the roles a document is shared with, and how
sensitive it is — and they need the rule to hold everywhere a document can
leak: the document list, a scoped query, a citation lookup, and retrieval.

The obvious way to add filtering to retrieval is to search, then drop the
rows the caller may not see. That is wrong in a way that is easy to miss.
FAISS and BM25 return the top N rows over the whole corpus. If forbidden rows
win most of those places, the caller's own relevant passages are crowded out,
and the answer gets worse — or refuses — because of documents the caller was
never allowed to know existed. Worse, the shape of the result (how many
survived the filter, which ones) is itself a signal about other tenants'
content.

## Decision

1. **One rule, one module.** `guardrails/access.py` decides visibility for
   documents and chunks: tenant first, then owner / tenant-admin / matching
   role (an empty role list means owner-only), then clearance against the
   ordered levels PUBLIC < INTERNAL < CONFIDENTIAL < RESTRICTED, then exact
   document scope. The registry, the retriever, the chunk endpoint and the
   query route all call it. A test fails if visibility logic appears anywhere
   else.

2. **Before scoring.** The retriever computes the allowed row set from the
   bound snapshot and hands it to FAISS as an `IDSelectorBatch` and to BM25 as
   a candidate mask. An excluded row is never scored, never ranked, never in
   agent state and never in a prompt.

3. **Denormalised onto the chunk.** `tenant_id`, `allowed_roles` and
   `classification` are copied from the registry record onto every chunk at
   ingestion, so retrieval needs no registry read. The registry record is the
   source of truth; a chunk is re-stamped whenever its document is re-indexed.

4. **A tripwire after.** The returned chunks are re-checked. With (2) in place
   this is a no-op; it exists so a regression in the selector path is logged
   loudly instead of becoming a leak. The retriever step records how many
   rows were excluded, so the trace shows the rule working.

5. **404, never 403**, for anything the caller may not see.

## Consequences

- The shared demo corpus (no tenant, no owner) stays visible to everyone,
  which is what makes a fresh clone useful.
- A tenant-wide document listing in AWS is a filtered Scan. At this scale it
  is cheap; a GSI on `tenant_id` is the upgrade when a tenant holds thousands
  of documents. Deliberately not provisioned on the free tier before then.
- Provenance (`provenance_hash`) is stamped in the same pass: sha256 over the
  file checksum, the character span and the text, so a reader can verify a
  citation without trusting the server.

**Tested by:** `backend/tests/test_rbac.py`, `backend/tests/test_provenance.py`.
