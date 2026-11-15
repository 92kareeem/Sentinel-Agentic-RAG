"""The document lifecycle as a state machine, and the dead-letter path.

Two writers racing on one document (an ingestion finishing while a delete
lands) must not both win, and a tombstoned document must never come back
because a slow job finished late. A publish that loses every race is not a
generic failure: it is a retryable dead letter, and retrying it must work.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from app.documents import registry
from app.models.schemas import DocumentRecord, DocumentStatus
from app.rag import embeddings, index_store, ingest_runtime

S = DocumentStatus
DEMO = {"x-api-key": "demo-local"}


def _seed(status: S, owner: str = "demo") -> DocumentRecord:
    now = registry.utc_now()
    return registry.put(
        DocumentRecord(
            document_id=registry.new_document_id(),
            owner_id=owner,
            tenant_id="acme",
            original_filename="p.md",
            safe_filename="p.md",
            content_type="text/markdown",
            file_size=10,
            checksum_sha256="x",
            storage_key="k",
            status=status,
            created_at=now,
            updated_at=now,
        )
    )


@pytest.mark.parametrize(
    ("start", "target", "ok"),
    [
        (S.UPLOADED, S.PROCESSING, True),
        (S.PROCESSING, S.INDEXED, True),
        (S.INDEXED, S.PROCESSING, True),  # re-index
        (S.FAILED, S.PROCESSING, True),  # retry
        (S.DELETED, S.PROCESSING, False),  # a tombstone stays dead
        (S.DELETED, S.INDEXED, False),
        (S.UPLOADED, S.INDEXED, False),  # cannot skip processing
    ],
)
def test_transitions_follow_the_state_machine(
    offline_env: Any, start: S, target: S, ok: bool
) -> None:
    rec = _seed(start)
    if ok:
        assert registry.update_status(rec.document_id, "demo", target).status == target
    else:
        with pytest.raises(registry.InvalidTransition):
            registry.update_status(rec.document_id, "demo", target)


def test_a_lost_race_is_detected_not_overwritten(offline_env: Any) -> None:
    rec = _seed(S.PROCESSING)
    stale = registry.get(rec.document_id, owner_id="demo")
    registry.update_status(rec.document_id, "demo", S.DELETED)  # another writer wins
    stale.status = S.INDEXED
    with pytest.raises(registry.InvalidTransition):
        registry.put(stale, expected_status=S.PROCESSING)
    assert registry.get(rec.document_id, owner_id="demo").status == S.DELETED


def test_processing_counts_ingest_attempts(offline_env: Any) -> None:
    rec = _seed(S.UPLOADED)
    registry.update_status(rec.document_id, "demo", S.PROCESSING)
    registry.update_status(rec.document_id, "demo", S.FAILED)
    registry.update_status(rec.document_id, "demo", S.PROCESSING)
    assert registry.get(rec.document_id, owner_id="demo").ingest_attempts == 2


def _cheap_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        embeddings, "embed_texts", lambda texts: np.ones((len(texts), 384), dtype=np.float32)
    )
    from app.rag.chunking import simple_word_offsets

    monkeypatch.setattr(embeddings, "token_offsets", simple_word_offsets)


def test_contention_dead_letters_the_document_and_retry_recovers_it(
    api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cheap_embeddings(monkeypatch)
    real = ingest_runtime._rebuild_and_publish

    def contended(*a: Any, **kw: Any) -> Any:
        raise index_store.ConcurrentUpdateError("lost every race")

    monkeypatch.setattr(ingest_runtime, "_rebuild_and_publish", contended)
    resp = api.post(
        "/v1/documents/local-upload",
        files={"file": ("p.md", b"# Policy\n\nRefunds take 30 days.", "text/markdown")},
        headers=DEMO,
    )
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "PUBLISH_CONTENTION"

    doc = api.get("/v1/documents", headers=DEMO).json()[0]
    assert doc["status"] == "FAILED" and doc["retryable"] is True
    assert doc["error_code"] == "PUBLISH_CONTENTION" and doc["ingest_attempts"] == 1

    monkeypatch.setattr(ingest_runtime, "_rebuild_and_publish", real)
    retried = api.post(f"/v1/documents/{doc['document_id']}/retry", headers=DEMO)
    assert retried.status_code == 200, retried.text
    doc = api.get(f"/v1/documents/{doc['document_id']}", headers=DEMO).json()
    assert doc["status"] == "INDEXED" and doc["ingest_attempts"] == 2 and doc["retryable"] is False


def test_a_permanent_failure_is_not_retryable(api: Any) -> None:
    rec = _seed(S.FAILED)
    registry.update_status(
        rec.document_id, "demo", S.FAILED, error_code="UNSUPPORTED_SCANNED_DOCUMENT"
    )  # type: ignore[arg-type]
    assert api.post(f"/v1/documents/{rec.document_id}/retry", headers=DEMO).status_code == 409


def test_deleting_removes_the_stored_upload(
    api: Any, monkeypatch: pytest.MonkeyPatch, offline_env: Any
) -> None:
    _cheap_embeddings(monkeypatch)
    doc_id = api.post(
        "/v1/documents/local-upload",
        files={"file": ("p.md", b"# Policy\n\nRefunds take 30 days.", "text/markdown")},
        headers=DEMO,
    ).json()["doc_id"]
    uploads = offline_env / "index" / "uploads"
    assert any(uploads.iterdir())
    assert api.delete(f"/v1/documents/{doc_id}", headers=DEMO).status_code == 200
    assert not any(uploads.iterdir())
