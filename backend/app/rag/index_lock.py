"""Cross-instance mutual exclusion for index publication.

THE BUG THIS FIXES
------------------
Publishing a document is a read-modify-write over the WHOLE index: load the
current version's chunks and vectors, merge one document in, write a new
version, move the pointer. Two writers running that concurrently can each read
version v10, each build their own v11, and each publish — and one user's
document silently disappears. No error is raised, nothing is logged, and the
document's registry row still says INDEXED. It is the worst shape of bug this
system can have.

ingest_runtime previously guarded that section with a `threading.RLock`, which
is correct for one process and useless across Lambda instances: two concurrent
invocations are two processes on two machines that share no memory. The
compare-and-set on the version pointer caught SOME of this, but only on a
shared filesystem — in AWS each Lambda has its own /tmp, so neither writer can
even see the other's pointer.

THE DESIGN
----------
A lease-based lock held as a single DynamoDB item, acquired with a conditional
write so acquisition is atomic at the database rather than in any process.

Deliberately stored in the EXISTING documents table under pk="LOCK#index"
rather than in a new table:

  * no new AWS resource to provision, and nothing new to pay for;
  * no IAM change — the role already has GetItem/PutItem/UpdateItem there;
  * the key layout (pk/sk strings) already fits, and "LOCK#" cannot collide
    with the registry's "USER#<id>" partitions.

RELEASE USES UpdateItem, NOT DeleteItem
---------------------------------------
The execution role deliberately has no dynamodb:DeleteItem (the application
does soft deletes everywhere). Rather than widen a least-privilege policy to
suit an implementation detail, releasing sets the lease to expired, which is
equivalent for every reader and needs only UpdateItem.

WHY A LEASE, AND WHY IT IS 120s
-------------------------------
A holder can die — Lambda times out, the instance is killed, the network
partitions — and a lock with no expiry would wedge ingestion permanently. The
lease bounds that: the lock self-heals once it passes.

The lease is sized to the FUNCTION TIMEOUT, not to typical work. A lease
shorter than the critical section is actively dangerous: a slow-but-healthy
holder would have its lock stolen mid-publish, which reintroduces exactly the
concurrent-writer bug this module exists to prevent. Paying up to one function
timeout of waiting after a crash is the cheaper side of that trade.

STILL NOT SUFFICIENT ALONE — AND WHY THE POINTER CAS STAYS
----------------------------------------------------------
A lease can expire while its holder is merely stalled (a long GC pause, a
throttled network call) rather than dead. That holder can wake up believing it
still owns the lock while a second writer also holds it. The compare-and-set in
index_store.publish() is the fence that catches this: the stalled writer's
base version is no longer current, so its publish is rejected with
ConcurrentUpdateError and retried against the new head instead of overwriting
it. Lock and CAS defend different failure modes; neither replaces the other.
"""

from __future__ import annotations

import random
import secrets
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app.config import get_settings

# Partition reserved for coordination rows. The registry only ever queries
# pk="USER#<owner_id>", so this row is invisible to every document read.
LOCK_PK = "LOCK#index"
LOCK_SK = "GLOBAL"


class IndexLockTimeout(RuntimeError):
    """Could not acquire the publication lock before the caller's deadline.

    Raised rather than proceeding unlocked: failing an upload loudly is
    recoverable, losing a previously-indexed document is not.
    """


# Serializes writers inside ONE process. Still required in AWS mode: without
# it, concurrent threads in a single container would fight over the distributed
# lock through the network instead of resolving locally and for free.
_process_lock = threading.RLock()


def _table() -> Any:
    import boto3

    settings = get_settings()
    return boto3.resource("dynamodb", region_name=settings.aws_region).Table(
        settings.ddb_table_documents
    )


def _try_acquire(token: str, lease_seconds: int) -> bool:
    """One atomic attempt. True if this caller now owns the lease.

    The condition is evaluated by DynamoDB at write time, so two callers racing
    with identical reads cannot both succeed — there is no read to be stale.
    """
    from botocore.exceptions import ClientError

    now = int(time.time())
    try:
        _table().put_item(
            Item={
                "pk": LOCK_PK,
                "sk": LOCK_SK,
                "holder": token,
                "expires_at": now + lease_seconds,
            },
            # Free if it has never been taken, or if the previous holder's lease
            # has run out. `attribute_not_exists(expires_at)` covers a
            # malformed row rather than wedging on it forever.
            ConditionExpression=(
                "attribute_not_exists(pk) "
                "OR attribute_not_exists(expires_at) "
                "OR expires_at < :now"
            ),
            ExpressionAttributeValues={":now": now},
        )
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return False
        raise


def _release(token: str) -> None:
    """Expire our own lease. Conditional on still being the holder.

    Without the condition, a caller whose lease had already expired and been
    taken by someone else would release THAT holder's lock, handing a third
    writer a lock two parties believe they own.
    """
    from botocore.exceptions import ClientError

    try:
        _table().update_item(
            Key={"pk": LOCK_PK, "sk": LOCK_SK},
            UpdateExpression="SET expires_at = :zero",
            ConditionExpression="holder = :token",
            ExpressionAttributeValues={":zero": 0, ":token": token},
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise
        # Our lease expired and another writer owns the lock. Releasing is not
        # ours to do, and the new holder's lease is what protects the index.


@contextmanager
def index_publication_lock() -> Iterator[None]:
    """Hold exclusive publication rights for the duration of the block.

    In local mode this is the in-process lock alone: there is exactly one
    writer, one filesystem, and nothing for a network round-trip to add.
    """
    settings = get_settings()

    if settings.local_mode:
        with _process_lock:
            yield
        return

    with _process_lock:
        token = secrets.token_hex(16)
        deadline = time.monotonic() + settings.index_lock_wait_seconds
        delay = 0.25

        while not _try_acquire(token, settings.index_lock_lease_seconds):
            if time.monotonic() >= deadline:
                raise IndexLockTimeout(
                    f"index publication lock held by another writer for more than "
                    f"{settings.index_lock_wait_seconds}s"
                )
            # Jittered backoff: several Lambdas that started together would
            # otherwise retry in lockstep forever.
            time.sleep(delay + random.uniform(0, 0.1))
            delay = min(delay * 2, 2.0)

        try:
            yield
        finally:
            _release(token)
