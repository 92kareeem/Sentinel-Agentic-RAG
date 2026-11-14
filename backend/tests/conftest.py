"""Shared fixtures for the HTTP-level suites.

Every test that builds the app gets a private INDEX_DIR, local storage, no
provider key (so nothing can reach a real LLM even if a developer's .env has
one), an empty answer cache and a fresh trace store.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from app.config import get_settings
from app.models.schemas import Chunk
from fastapi.testclient import TestClient


@pytest.fixture()
def offline_env(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("INDEX_REFRESH_SECONDS", "0")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("REGRESSIONS_PATH", str(tmp_path / "regressions.json"))
    get_settings.cache_clear()

    from app.learning import casebook
    from app.observability import tracing
    from app.rag import answer_cache, retriever_snapshot

    retriever_snapshot.invalidate()
    answer_cache.clear()
    tracing.reset_local()
    casebook.reset_local()
    yield tmp_path
    retriever_snapshot.invalidate()
    answer_cache.clear()
    tracing.reset_local()
    get_settings.cache_clear()


def make_client(monkeypatch: pytest.MonkeyPatch, *, demo: bool = False) -> TestClient:
    if demo:
        monkeypatch.setenv("DEMO_MODE", "true")
        get_settings.cache_clear()
    from app.main import create_app

    return TestClient(create_app())


@pytest.fixture()
def api(offline_env: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    with make_client(monkeypatch) as c:
        yield c


def publish_chunks(chunks: list[Chunk]) -> str:
    """Publish a tiny index so a snapshot has a version (the cache keys on it)."""
    from app.rag import bm25_store, index_store, retriever_snapshot

    root = get_settings().index_dir
    vecs = np.ones((len(chunks), 384), dtype=np.float32)
    version = index_store.publish(
        index_store.build_faiss(vecs),
        chunks,
        bm25_store.build([c.embed_text for c in chunks]),
        base_version=index_store.read_pointer(root),
        root=root,
    )
    retriever_snapshot.invalidate()
    return version


def chunk(cid: str, text: str = "Refunds take 3 days.") -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id="refund-policy",
        section_path="Refunds",
        text=text,
        token_count=5,
        char_start=0,
        char_end=len(text),
    )
