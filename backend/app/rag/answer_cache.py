"""Verified-answer cache: never pay twice for an answer already proven.

Role in architecture: an answer that passed the critic AND the deterministic
grounding gate, for this exact question, against this exact index version,
for a caller with this exact visibility, is a fact the system already
established. Serving it again cannot be less grounded than the first time,
and it costs zero tokens and a few milliseconds.

What makes that safe is the key, not the TTL:

  * index version   any publish (upload, delete, re-index) changes it, so a
                    corpus change invalidates every entry by construction;
                    there is no invalidation path to forget
  * user + tenant   visibility differs per caller (owner-only documents,
  * roles/clearance role grants, classification), so an answer built from
                    evidence one user may see is never served to another
  * scope           a document-scoped question is a different question
  * provider/model  a model switch is a deliberate quality change
  * question        normalised for case and whitespace only — semantic
                    matching would serve one question's answer to another

Never cached: refusals (a BUDGET_EXHAUSTED refusal is transient, and a gap
should be re-tested once someone uploads the missing document) and anything
asked with conversation history (a follow-up means something different in a
different conversation).

Process-local by design. On Lambda each warm container has its own cache;
a shared cache would need ElastiCache, which is not free tier, and the hit
rate of a per-container cache is reported honestly in the telemetry tab.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from app.config import get_settings
from app.models.schemas import AccessContext, QueryResponse


@dataclass(frozen=True)
class CacheKey:
    digest: str


def _normalise(question: str) -> str:
    return " ".join(question.lower().split())


def make_key(
    *, question: str, ctx: AccessContext, doc_id: str | None, index_version: str | None
) -> CacheKey | None:
    """None when there is nothing safe to key on (no published index)."""
    if index_version is None:
        return None
    settings = get_settings()
    parts = [
        index_version,
        ctx.tenant_id,
        ctx.user_id,
        ",".join(sorted(ctx.roles)),
        ctx.clearance.value,
        doc_id or "*",
        settings.llm_provider,
        settings.model_simple,
        _normalise(question),
    ]
    return CacheKey(hashlib.sha256("\x1f".join(parts).encode()).hexdigest())


class _LRU:
    def __init__(self) -> None:
        self._data: OrderedDict[str, tuple[float, QueryResponse]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: CacheKey) -> QueryResponse | None:
        ttl = get_settings().answer_cache_ttl_seconds
        if ttl <= 0:
            return None
        with self._lock:
            hit = self._data.get(key.digest)
            if hit is None:
                return None
            stored_at, response = hit
            if time.monotonic() - stored_at > ttl:
                del self._data[key.digest]
                return None
            self._data.move_to_end(key.digest)
            return response

    def put(self, key: CacheKey, response: QueryResponse) -> None:
        settings = get_settings()
        if settings.answer_cache_ttl_seconds <= 0:
            return
        with self._lock:
            self._data[key.digest] = (time.monotonic(), response)
            self._data.move_to_end(key.digest)
            while len(self._data) > settings.answer_cache_max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


_cache = _LRU()


def lookup(key: CacheKey | None) -> QueryResponse | None:
    return None if key is None else _cache.get(key)


def store(key: CacheKey | None, response: QueryResponse) -> None:
    if key is not None:
        _cache.put(key, response)


def clear() -> None:
    _cache.clear()


def size() -> int:
    return len(_cache)
