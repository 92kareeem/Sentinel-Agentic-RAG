"""Ingestion lifecycle: a document's registry state must always reflect reality.

Replaces the previous test of submit_ingest_job()/get_job_status(), which
covered an in-process daemon-thread "job system". That design was removed
rather than tested further: on Lambda the runtime freezes when the handler
returns, so a queued job could simply never run and the document would sit in
PROCESSING forever. What matters now is that ingestion is transactional
against the registry — every path ends in a terminal, inspectable state.
"""

import pytest
from app.config import get_settings
from app.documents import registry
from app.models.schemas import DocumentErrorCode, DocumentRecord, DocumentStatus
from app.rag import ingest_runtime
from app.rag.pdf import PdfExtractionError


@pytest.fixture(autouse=True)
def _isolated_index(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _register(doc_id: str = "doc-1", owner: str = "user-1") -> DocumentRecord:
    now = registry.utc_now()
    return registry.put(
        DocumentRecord(
            document_id=doc_id,
            owner_id=owner,
            original_filename="doc.txt",
            safe_filename="doc.txt",
            content_type="text/plain",
            file_size=11,
            checksum_sha256="x",
            storage_key="local/user-1/doc.txt",
            status=DocumentStatus.UPLOADED,
            created_at=now,
            updated_at=now,
        )
    )


def test_successful_ingestion_marks_indexed_with_provenance(tmp_path, monkeypatch) -> None:
    record = _register()
    path = tmp_path / "doc.txt"
    path.write_text("# Title\n\nRefunds take 30 days to process.", encoding="utf-8")

    monkeypatch.setattr(
        ingest_runtime, "_rebuild_and_publish", lambda root, doc_id, chunks: (len(chunks), "v7")
    )

    count, version = ingest_runtime.ingest_document(
        path, document_id=record.document_id, owner_id=record.owner_id, source_filename="doc.txt"
    )
    assert count > 0 and version == "v7"

    stored = registry.get(record.document_id, owner_id=record.owner_id)
    assert stored.status == DocumentStatus.INDEXED
    assert stored.chunk_count == count
    assert stored.index_version == "v7"
    assert stored.embedding_model  # provenance recorded for reproducibility
    assert stored.error_code is None


def test_extraction_failure_marks_failed_with_reason(tmp_path, monkeypatch) -> None:
    """A PDF we cannot read must FAIL loudly, not index as an empty success."""
    record = _register(doc_id="doc-bad")
    path = tmp_path / "doc.txt"
    path.write_text("ignored", encoding="utf-8")

    def boom(*a, **kw):
        raise PdfExtractionError(
            DocumentErrorCode.ENCRYPTED_DOCUMENT, "PDF is password-protected"
        )

    monkeypatch.setattr(ingest_runtime, "chunk_file", boom)

    with pytest.raises(PdfExtractionError):
        ingest_runtime.ingest_document(
            path,
            document_id=record.document_id,
            owner_id=record.owner_id,
            source_filename="doc.txt",
        )

    stored = registry.get(record.document_id, owner_id=record.owner_id)
    assert stored.status == DocumentStatus.FAILED
    assert stored.error_code == DocumentErrorCode.ENCRYPTED_DOCUMENT
    assert "password" in stored.error_message


def test_unexpected_error_still_leaves_terminal_state(tmp_path, monkeypatch) -> None:
    """No exception path may leave a document stuck in PROCESSING."""
    record = _register(doc_id="doc-boom")
    path = tmp_path / "doc.txt"
    path.write_text("# T\n\nsome text", encoding="utf-8")

    def boom(*a, **kw):
        raise RuntimeError("index exploded")

    monkeypatch.setattr(ingest_runtime, "_rebuild_and_publish", boom)

    with pytest.raises(RuntimeError):
        ingest_runtime.ingest_document(
            path,
            document_id=record.document_id,
            owner_id=record.owner_id,
            source_filename="doc.txt",
        )

    stored = registry.get(record.document_id, owner_id=record.owner_id)
    assert stored.status == DocumentStatus.FAILED
    assert stored.error_code == DocumentErrorCode.INTERNAL_ERROR
