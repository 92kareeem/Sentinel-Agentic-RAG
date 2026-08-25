"""THE acceptance test: the business goal, end to end, deterministically.

Covers the full journey — upload a real <1MB PDF, have it registered, parsed,
chunked, embedded, indexed and marked INDEXED, then answer questions about
content on specific pages and inside a table, refuse questions the document
cannot support, and refuse another tenant access to it.

The LLM is stubbed (synthesizer/critic/router), because what is under test here
is the PRODUCT PIPELINE — identity, persistence, retrieval, scoping,
authorization, grounding, refusal. Live-model answer quality is measured
separately by evals/run.py, which runs against real Groq. Everything else in
this test is the genuine code path: real PyMuPDF extraction, real chunking,
real MiniLM embeddings, real FAISS + BM25 retrieval, real index publication.
"""

from __future__ import annotations

import pytest
from app.agents import retriever
from app.config import get_settings
from app.documents import registry
from app.models.schemas import DocumentStatus
from app.rag import index_store
from app.rag.ingest_runtime import delete_document, ingest_document

from . import pdf_fixtures as fx

OWNER_A = "user-alice"
OWNER_B = "user-bob"


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Every test gets a private index + registry so ordering cannot matter."""
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("LOCAL_MODE", "true")
    get_settings.cache_clear()
    retriever.reset_cache()
    yield
    get_settings.cache_clear()
    retriever.reset_cache()


def _register(owner: str, filename: str, path) -> str:
    """Mirror what the upload route records before ingestion begins."""
    import hashlib

    contents = path.read_bytes()
    now = registry.utc_now()
    from app.models.schemas import DocumentRecord

    record = registry.put(
        DocumentRecord(
            document_id=registry.new_document_id(),
            owner_id=owner,
            original_filename=filename,
            safe_filename=filename,
            content_type="application/pdf",
            file_size=len(contents),
            checksum_sha256=hashlib.sha256(contents).hexdigest(),
            storage_key=f"local/{owner}/{filename}",
            status=DocumentStatus.UPLOADED,
            created_at=now,
            updated_at=now,
        )
    )
    return record.document_id


def _retrieve(query: str, *, owner: str, doc_id: str | None) -> list:
    """Drive the real retriever node (hybrid dense+sparse, RRF, rerank)."""
    import time

    from app.agents.state import AgentState
    from app.observability.tracing import TraceRecorder

    state: AgentState = {
        "query": query,
        "user_id": owner,
        "doc_id": doc_id,
        "trace": TraceRecorder(),
        "attempt": 0,
        "model": "stub",
        "token_budget_left": 10_000,
        "deadline_ts": time.monotonic() + 60,
        "retrieved": [],
        "answer": "",
        "citations": [],
        "critic": None,
        "status": "running",
        "conversation_history": [],
    }
    retriever.retriever_node(state)
    return state["retrieved"]


def test_business_goal_pdf_to_grounded_answer(tmp_path) -> None:
    """1-17 of the acceptance criteria, in order."""
    # 1. a representative PDF, comfortably under the 1 MB limit
    pdf_path = fx.multipage_pdf(tmp_path / "handbook.pdf")
    assert 0 < pdf_path.stat().st_size < get_settings().max_upload_bytes

    # 2-3. uploaded and registered under a server-generated identity
    doc_id = _register(OWNER_A, "handbook.pdf", pdf_path)
    record = registry.get(doc_id, owner_id=OWNER_A)
    assert record is not None
    assert record.status == DocumentStatus.UPLOADED
    assert record.document_id not in ("handbook", "handbook.pdf")  # never the filename

    # 4-5. processing runs to completion
    chunk_count, version = ingest_document(
        pdf_path, document_id=doc_id, owner_id=OWNER_A, source_filename="handbook.pdf"
    )
    assert chunk_count > 0

    # 6-7. chunks persisted under a valid, published index version
    indexed = registry.get(doc_id, owner_id=OWNER_A)
    assert indexed.status == DocumentStatus.INDEXED
    assert indexed.chunk_count == chunk_count
    assert indexed.index_version == version
    assert indexed.page_count == 3
    assert indexed.embedding_model and indexed.parser_version  # provenance

    root = tmp_path / "index"
    assert index_store.read_pointer(root) == version
    manifest = index_store.read_manifest(version, root)
    assert manifest.chunk_count == chunk_count
    assert doc_id in manifest.document_ids

    retriever.reset_cache()

    # 8-10. a question answered from page 1, cited to the right document + page
    hits = _retrieve("What is the reimbursement window?", owner=OWNER_A, doc_id=doc_id)
    assert hits, "expected retrieval hits for a fact that is in the document"
    assert all(h.doc_id == doc_id for h in hits)
    top = hits[0]
    assert "45 days" in top.text
    assert top.page_number == 1
    assert top.source_filename == "handbook.pdf"

    # 11-12. a question whose answer lives inside a table
    table_hits = _retrieve(
        "What laptop budget does a Designer receive?", owner=OWNER_A, doc_id=doc_id
    )
    joined = " ".join(h.text for h in table_hits)
    assert fx.TABLE_FACT_BUDGET in joined
    assert fx.TABLE_FACT_ROLE in joined

    # 13-14. a fact from a later page is still reachable (whole doc is indexed,
    # not just the first page)
    late = _retrieve("Does unused allowance roll over?", owner=OWNER_A, doc_id=doc_id)
    assert any("roll over" in h.text for h in late)
    assert any(h.page_number == 3 for h in late)

    # 15-16. a question the document cannot support retrieves nothing relevant.
    # With doc scope + no matching content, the synthesizer receives evidence
    # that does not contain the answer; grounding/refusal is asserted directly
    # in test_refusal_when_evidence_absent below.
    absent = _retrieve(f"What is the {fx.ABSENT_TOPIC} policy?", owner=OWNER_A, doc_id=doc_id)
    assert not any(fx.ABSENT_TOPIC in h.text.lower() for h in absent)

    # 17. another user cannot reach this document
    assert registry.get(doc_id, owner_id=OWNER_B) is None
    assert _retrieve("reimbursement window", owner=OWNER_B, doc_id=doc_id) == []


def test_same_filename_two_users_stay_isolated(tmp_path) -> None:
    """The exact collision the old filename-derived identity could not survive."""
    a_pdf = fx.simple_text_pdf(tmp_path / "a" / "policy.pdf")
    b_pdf = fx.multipage_pdf(tmp_path / "b" / "policy.pdf")

    a_id = _register(OWNER_A, "policy.pdf", a_pdf)
    b_id = _register(OWNER_B, "policy.pdf", b_pdf)
    assert a_id != b_id

    ingest_document(a_pdf, document_id=a_id, owner_id=OWNER_A, source_filename="policy.pdf")
    ingest_document(b_pdf, document_id=b_id, owner_id=OWNER_B, source_filename="policy.pdf")
    retriever.reset_cache()

    # both survived: neither overwrote the other in the index
    assert registry.get(a_id, owner_id=OWNER_A).status == DocumentStatus.INDEXED
    assert registry.get(b_id, owner_id=OWNER_B).status == DocumentStatus.INDEXED

    # each user's unscoped search sees only their own document
    a_hits = _retrieve("access requests OPS-204", owner=OWNER_A, doc_id=None)
    assert all(h.doc_id == a_id for h in a_hits)
    b_hits = _retrieve("reimbursement window", owner=OWNER_B, doc_id=None)
    assert all(h.doc_id == b_id for h in b_hits)

    # and cross-tenant scoping yields nothing
    assert _retrieve("anything", owner=OWNER_A, doc_id=b_id) == []


def test_reupload_replaces_rather_than_duplicates(tmp_path) -> None:
    pdf_path = fx.multipage_pdf(tmp_path / "handbook.pdf")
    doc_id = _register(OWNER_A, "handbook.pdf", pdf_path)

    first, _ = ingest_document(
        pdf_path, document_id=doc_id, owner_id=OWNER_A, source_filename="handbook.pdf"
    )
    second, version = ingest_document(
        pdf_path, document_id=doc_id, owner_id=OWNER_A, source_filename="handbook.pdf"
    )
    assert first == second

    _, chunks, _ = index_store.load_current(tmp_path / "index")
    assert sum(1 for c in chunks if c.doc_id == doc_id) == second  # not doubled


def test_delete_removes_content_from_retrieval(tmp_path) -> None:
    pdf_path = fx.multipage_pdf(tmp_path / "handbook.pdf")
    doc_id = _register(OWNER_A, "handbook.pdf", pdf_path)
    ingest_document(
        pdf_path, document_id=doc_id, owner_id=OWNER_A, source_filename="handbook.pdf"
    )
    retriever.reset_cache()
    assert _retrieve("reimbursement window", owner=OWNER_A, doc_id=None)

    delete_document(doc_id, OWNER_A)
    retriever.reset_cache()

    assert registry.get(doc_id, owner_id=OWNER_A).status == DocumentStatus.DELETED
    assert _retrieve("reimbursement window", owner=OWNER_A, doc_id=None) == []
    assert registry.list_for_owner(OWNER_A) == []


def test_failed_document_is_never_reported_as_indexed(tmp_path) -> None:
    """A scanned PDF must land in FAILED with a reason, not a silent success."""
    from app.rag.pdf import PdfExtractionError

    bad = fx.empty_pdf(tmp_path / "scanned.pdf")
    doc_id = _register(OWNER_A, "scanned.pdf", bad)

    with pytest.raises(PdfExtractionError):
        ingest_document(
            bad, document_id=doc_id, owner_id=OWNER_A, source_filename="scanned.pdf"
        )

    record = registry.get(doc_id, owner_id=OWNER_A)
    assert record.status == DocumentStatus.FAILED
    assert record.error_code.value == "UNSUPPORTED_SCANNED_DOCUMENT"
    assert record.chunk_count == 0


def test_refusal_when_evidence_absent(tmp_path) -> None:
    """Grounding must reject an answer whose claims aren't in the evidence."""
    from app.guardrails import grounding
    from app.models.schemas import Chunk

    evidence = [
        Chunk(
            chunk_id="c1",
            doc_id="d1",
            section_path="Reimbursement",
            text=fx.FACT_PAGE_1,
            token_count=10,
            char_start=0,
            char_end=len(fx.FACT_PAGE_1),
        )
    ]
    fabricated = "Parental leave is 16 weeks fully paid for all employees [chunk:c1]."
    result = grounding.verify(fabricated, evidence)
    assert not result.ok, "a claim unsupported by its evidence must not ship"
