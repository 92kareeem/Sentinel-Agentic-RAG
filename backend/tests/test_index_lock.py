"""The distributed publication lock.

These tests target the DISTRIBUTED semantics specifically — the in-process
threading lock was never the problem. Concurrency between Lambda instances
cannot be reproduced with threads (two threads share the process lock and
never reach DynamoDB), so the conditional-write primitives are driven directly
with two different holder tokens, which is exactly what two instances are.
"""

from __future__ import annotations

import time

import boto3
import pytest
from app.config import get_settings
from app.documents import registry
from app.models.schemas import DocumentRecord, DocumentStatus
from app.rag import index_lock
from moto import mock_aws


@pytest.fixture()
def ddb_documents(monkeypatch: pytest.MonkeyPatch):
    """A moto-backed documents table — the same table the lock row lives in."""
    monkeypatch.setenv("LOCAL_MODE", "false")
    get_settings.cache_clear()
    with mock_aws():
        settings = get_settings()
        boto3.resource("dynamodb", region_name=settings.aws_region).create_table(
            TableName=settings.ddb_table_documents,
            KeySchema=[
                {"AttributeName": "pk", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        yield
    get_settings.cache_clear()


# ------------------------------------------------------- the bug being fixed


def test_two_writers_cannot_hold_the_lock_at_once(ddb_documents: None) -> None:
    """THE regression test. Two Lambda instances publishing concurrently each
    read the same base version, each build a new one, and one document is lost
    with no error raised anywhere. The second writer must be refused."""
    assert index_lock._try_acquire("instance-a", 120) is True
    assert index_lock._try_acquire("instance-b", 120) is False


def test_lock_is_released_for_the_next_writer(ddb_documents: None) -> None:
    assert index_lock._try_acquire("instance-a", 120) is True
    index_lock._release("instance-a")
    assert index_lock._try_acquire("instance-b", 120) is True


def test_a_dead_holders_lease_expires_so_the_lock_cannot_wedge(
    ddb_documents: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A holder can die mid-publish (Lambda timeout, instance killed). Without
    lease expiry the index would be unpublishable forever."""
    assert index_lock._try_acquire("dead-writer", 120) is True
    assert index_lock._try_acquire("new-writer", 120) is False

    real_time = time.time
    monkeypatch.setattr(index_lock.time, "time", lambda: real_time() + 121)

    assert index_lock._try_acquire("new-writer", 120) is True


def test_release_cannot_steal_the_lock_from_the_current_holder(
    ddb_documents: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writer whose lease expired must not release the lock that a DIFFERENT
    writer has since acquired — that would hand a third writer a lock two
    parties believe they hold."""
    assert index_lock._try_acquire("slow-writer", 120) is True

    real_time = time.time
    monkeypatch.setattr(index_lock.time, "time", lambda: real_time() + 121)
    assert index_lock._try_acquire("new-holder", 120) is True

    # The stalled writer wakes up and tries to tidy up after itself.
    index_lock._release("slow-writer")

    # new-holder must still own it.
    assert index_lock._try_acquire("third-writer", 120) is False


def test_waiting_writer_gives_up_loudly_rather_than_publishing_unlocked(
    ddb_documents: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failing an upload is recoverable; publishing without the lock can delete
    a previously-indexed document."""
    monkeypatch.setenv("INDEX_LOCK_WAIT_SECONDS", "1")
    get_settings.cache_clear()

    assert index_lock._try_acquire("someone-else", 120) is True

    with pytest.raises(index_lock.IndexLockTimeout), index_lock.index_publication_lock():
        pytest.fail("acquired a lock that another writer holds")


# ------------------------------------------------------- coexistence with data


def test_lock_row_is_invisible_to_the_document_registry(ddb_documents: None) -> None:
    """The lock shares the documents table to avoid provisioning a second one.
    It must never surface as a document."""
    now = registry.utc_now()
    registry.put(
        DocumentRecord(
            document_id="d1",
            owner_id="alice",
            original_filename="a.pdf",
            safe_filename="a.pdf",
            content_type="application/pdf",
            file_size=10,
            checksum_sha256="x",
            storage_key="uploads/alice/d1/a.pdf",
            status=DocumentStatus.INDEXED,
            created_at=now,
            updated_at=now,
        )
    )
    assert index_lock._try_acquire("instance-a", 120) is True

    docs = registry.list_for_owner("alice")
    assert [d.document_id for d in docs] == ["d1"]


# ------------------------------------------------------------------ local mode


def test_local_mode_uses_the_process_lock_and_needs_no_aws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_MODE", "true")
    get_settings.cache_clear()
    try:
        # Nested on purpose: the lock must be reentrant, because a delete
        # publishes a new version while the caller already holds it.
        with index_lock.index_publication_lock(), index_lock.index_publication_lock():
            pass
    finally:
        get_settings.cache_clear()
