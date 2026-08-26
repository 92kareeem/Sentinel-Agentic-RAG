"""HTTP-level API tests through the real FastAPI app.

These exist because the unit suite drives nodes and modules directly and
therefore missed a whole class of defect: the document-scope authorization
check returned 500 instead of 404 because a function-local
`from fastapi import HTTPException` shadowed the module-level import, making
every earlier reference an UnboundLocalError. Nothing below mocks the app —
only the LLM is stubbed.
"""

from __future__ import annotations

import pytest
from app.config import get_settings
from app.documents import registry
from app.models.schemas import DocumentRecord, DocumentStatus
from fastapi.testclient import TestClient

DEMO = {"x-api-key": "demo-local"}
ADMIN = {"x-api-key": "admin-local"}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_MODE", "true")
    get_settings.cache_clear()

    from app.agents import retriever
    from app.main import create_app

    retriever.reset_cache()
    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()
    retriever.reset_cache()


def _seed(owner: str, status: DocumentStatus = DocumentStatus.INDEXED) -> str:
    now = registry.utc_now()
    record = registry.put(
        DocumentRecord(
            document_id=registry.new_document_id(),
            owner_id=owner,
            original_filename="doc.pdf",
            safe_filename="doc.pdf",
            content_type="application/pdf",
            file_size=10,
            checksum_sha256="x",
            storage_key=f"local/{owner}/doc.pdf",
            status=status,
            created_at=now,
            updated_at=now,
        )
    )
    return record.document_id


# ---------------------------------------------------------------- auth


def test_missing_api_key_is_rejected(client) -> None:
    """No key -> rejected before any work. 400, not FastAPI's default 422:
    this app maps every RequestValidationError into its RFC 7807 shape."""
    assert client.post("/v1/query", json={"query": "hi"}).status_code == 400
    assert client.get("/v1/documents").status_code == 400


def test_unknown_api_key_is_rejected(client) -> None:
    resp = client.get("/v1/documents", headers={"x-api-key": "nope"})
    assert resp.status_code == 401


# ---------------------------------------------------------------- documents


def test_documents_list_is_scoped_to_the_caller(client) -> None:
    _seed("demo")
    _seed("admin")
    body = client.get("/v1/documents", headers=DEMO).json()
    assert len(body) == 1
    assert all(d["document_id"] for d in body)


def test_other_users_document_is_404_not_403(client) -> None:
    """404 rather than 403: a 403 would confirm the id exists."""
    doc_id = _seed("admin")
    assert client.get(f"/v1/documents/{doc_id}", headers=DEMO).status_code == 404


def test_scoped_query_on_foreign_document_is_404(client) -> None:
    """Regression: this returned 500 (UnboundLocalError), not 404."""
    doc_id = _seed("admin")
    resp = client.post("/v1/query", json={"query": "secrets?", "doc_id": doc_id}, headers=DEMO)
    assert resp.status_code == 404, resp.text


def test_scoped_query_on_unknown_document_is_404(client) -> None:
    resp = client.post(
        "/v1/query", json={"query": "x", "doc_id": "does-not-exist"}, headers=DEMO
    )
    assert resp.status_code == 404


def test_query_on_document_still_processing_is_409(client) -> None:
    """A not-yet-queryable document must say so, not answer from an empty index."""
    doc_id = _seed("demo", status=DocumentStatus.PROCESSING)
    resp = client.post("/v1/query", json={"query": "x", "doc_id": doc_id}, headers=DEMO)
    assert resp.status_code == 409
    assert "PROCESSING" in resp.text


# ---------------------------------------------------------------- uploads


def test_upload_rejects_non_pdf_content_claiming_to_be_pdf(client) -> None:
    """The extension is a client claim; the bytes are checked."""
    resp = client.post(
        "/v1/documents/local-upload",
        files={"file": ("evil.pdf", b"this is not a pdf", "application/pdf")},
        headers=DEMO,
    )
    assert resp.status_code == 400


def test_upload_rejects_empty_file(client) -> None:
    resp = client.post(
        "/v1/documents/local-upload",
        files={"file": ("empty.txt", b"", "text/plain")},
        headers=DEMO,
    )
    assert resp.status_code == 400


def test_upload_rejects_oversize_file(client) -> None:
    limit = get_settings().max_upload_bytes
    resp = client.post(
        "/v1/documents/local-upload",
        files={"file": ("big.txt", b"x" * (limit + 1), "text/plain")},
        headers=DEMO,
    )
    assert resp.status_code == 413


def test_upload_rejects_unsupported_extension(client) -> None:
    resp = client.post(
        "/v1/documents/local-upload",
        files={"file": ("payload.exe", b"MZ...", "application/octet-stream")},
        headers=DEMO,
    )
    assert resp.status_code == 400


def test_path_traversal_filename_is_sanitized(client) -> None:
    """A traversal filename must not escape the storage prefix."""
    from app.api.routes_ingest import _safe_filename

    safe = _safe_filename("../../../../etc/passwd.txt")
    assert "/" not in safe and ".." not in safe
    assert safe.endswith(".txt")


def test_scanned_pdf_upload_reports_actionable_error(client, tmp_path) -> None:
    from . import pdf_fixtures as fx

    blank = fx.empty_pdf(tmp_path / "scanned.pdf")
    resp = client.post(
        "/v1/documents/local-upload",
        files={"file": ("scanned.pdf", blank.read_bytes(), "application/pdf")},
        headers=DEMO,
    )
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "UNSUPPORTED_SCANNED_DOCUMENT"
    assert "scanned" in resp.json()["detail"].lower()


# ---------------------------------------------------------------- traces


def test_trace_of_another_user_is_forbidden(client) -> None:
    from app.observability.tracing import put_trace

    put_trace({"trace_id": "t-1", "user_id": "admin", "steps": []})
    assert client.get("/v1/traces/t-1", headers=DEMO).status_code == 403
    assert client.get("/v1/traces/t-1", headers=ADMIN).status_code == 200


def test_trace_listing_is_admin_only(client) -> None:
    assert client.get("/v1/traces", headers=DEMO).status_code == 403
    assert client.get("/v1/traces", headers=ADMIN).status_code == 200
