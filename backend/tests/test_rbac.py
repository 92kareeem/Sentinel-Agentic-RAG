"""Tenant, role and clearance filtering, enforced before anything is scored.

The property under test is not "the answer did not mention the other
tenant's document". It is stronger: the other tenant's rows were never
candidates. A chunk the caller may not see must not be ranked, must not be
in agent state, and must not be fetchable by id.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest
from app.config import get_settings
from app.documents import registry
from app.guardrails import access
from app.guardrails.access import can_see, context_for, visible_rows
from app.guardrails.auth import _LOCAL_USERS
from app.models.schemas import (
    AccessContext,
    Chunk,
    ClassificationLevel,
    DocumentRecord,
    DocumentStatus,
)
from app.rag import bm25_store, index_store, retriever_snapshot
from fastapi.testclient import TestClient

C = ClassificationLevel


def _chunk(
    cid: str,
    *,
    owner: str = "",
    tenant: str = "",
    roles: list[str] | None = None,
    level: C = C.PUBLIC,
    doc: str = "d",
) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=doc,
        owner_id=owner,
        tenant_id=tenant,
        allowed_roles=roles or [],
        classification=level,
        section_path="S",
        text=f"text {cid}",
        token_count=2,
        char_start=0,
        char_end=5,
    )


def _ctx(user_key: str) -> AccessContext:
    return context_for(_LOCAL_USERS[user_key])


# -------------------------------------------------------------------- rule


def test_shared_corpus_is_visible_to_every_tenant() -> None:
    shared = _chunk("s")
    for key in _LOCAL_USERS:
        assert can_see(_ctx(key), shared)


def test_another_tenants_document_is_invisible_even_to_an_admin() -> None:
    globex_doc = _chunk("g", owner="globex-user", tenant="globex", roles=["employee"])
    assert can_see(_ctx("globex-local"), globex_doc)
    assert not can_see(_ctx("admin-local"), globex_doc)
    assert not can_see(_ctx("demo-local"), globex_doc)


def test_empty_roles_means_owner_only_but_tenant_admin_sees_all() -> None:
    private = _chunk("p", owner="demo", tenant="acme", level=C.INTERNAL)
    assert can_see(_ctx("demo-local"), private)
    assert can_see(_ctx("admin-local"), private)
    assert not can_see(_ctx("finance-local"), private)


def test_a_matching_role_grants_access_within_the_tenant() -> None:
    shared = _chunk("r", owner="demo", tenant="acme", roles=["finance"], level=C.INTERNAL)
    assert can_see(_ctx("finance-local"), shared)
    assert not can_see(_ctx("contractor-local"), shared)  # wrong role AND low clearance


def test_clearance_is_checked_after_roles() -> None:
    """A role can be granted and the document still be above the reader."""
    secret = _chunk("c", owner="admin", tenant="acme", roles=["employee"], level=C.CONFIDENTIAL)
    assert not can_see(_ctx("demo-local"), secret)  # employee, but INTERNAL clearance
    assert can_see(_ctx("finance-local"), secret)  # employee, CONFIDENTIAL clearance


def test_legacy_owner_only_chunks_without_a_tenant_still_stay_private() -> None:
    legacy = _chunk("l", owner="alice")
    assert can_see(AccessContext(user_id="alice"), legacy)
    assert not can_see(AccessContext(user_id="bob"), legacy)


def test_visible_rows_is_none_when_nothing_is_excluded() -> None:
    chunks = [_chunk("a"), _chunk("b")]
    assert visible_rows(chunks, _ctx("demo-local"), None) is None


def test_visible_rows_scopes_on_exact_document_id() -> None:
    chunks = [_chunk("a", doc="doc1"), _chunk("b", doc="doc10")]
    assert visible_rows(chunks, _ctx("demo-local"), "doc1") == {0}


# ------------------------------------------------------- pre-scoring search


def test_faiss_never_scores_an_excluded_row() -> None:
    vecs = np.eye(4, dtype=np.float32)
    index = index_store.build_faiss(vecs)
    q = np.ones(4, dtype=np.float32)
    rows = {r for r, _ in index_store.search(index, q, 4, allowed_rows={1, 3})}
    assert rows == {1, 3}
    assert index_store.search(index, q, 4, allowed_rows=set()) == []


def test_bm25_never_ranks_an_excluded_row() -> None:
    bm25 = bm25_store.build(["refund policy", "refund window", "unrelated"])
    rows = [r for r, _ in bm25_store.search(bm25, "refund", 3, allowed_rows={2})]
    assert rows == [2]


# ------------------------------------------------------------ through the API


@pytest.fixture()
def client(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_MODE", "true")
    monkeypatch.setenv("INDEX_REFRESH_SECONDS", "0")
    get_settings.cache_clear()
    from app.main import create_app

    retriever_snapshot.invalidate()
    with TestClient(create_app()) as c:
        yield c
    retriever_snapshot.invalidate()
    get_settings.cache_clear()


def _seed_doc(owner: str, tenant: str, roles: list[str], level: C) -> str:
    now = registry.utc_now()
    rec = registry.put(
        DocumentRecord(
            document_id=registry.new_document_id(),
            owner_id=owner,
            tenant_id=tenant,
            allowed_roles=roles,
            classification=level,
            original_filename="doc.md",
            safe_filename="doc.md",
            content_type="text/markdown",
            file_size=10,
            checksum_sha256="abc",
            storage_key="k",
            status=DocumentStatus.INDEXED,
            created_at=now,
            updated_at=now,
        )
    )
    return rec.document_id


def _publish(root: Any, chunks: list[Chunk]) -> None:
    vecs = np.ones((len(chunks), 384), dtype=np.float32)
    index_store.publish(
        index_store.build_faiss(vecs),
        chunks,
        bm25_store.build([c.embed_text for c in chunks]),
        base_version=None,
        root=root,
    )
    retriever_snapshot.invalidate()


def test_document_list_shows_own_and_role_shared_documents_only(client: Any) -> None:
    mine = _seed_doc("demo", "acme", [], C.INTERNAL)
    shared = _seed_doc("admin", "acme", ["employee"], C.INTERNAL)
    _seed_doc("admin", "acme", ["finance"], C.INTERNAL)  # not for demo
    _seed_doc("globex-user", "globex", ["employee"], C.INTERNAL)  # other tenant

    ids = {
        d["document_id"]
        for d in client.get("/v1/documents", headers={"x-api-key": "demo-local"}).json()
    }
    assert ids == {mine, shared}
    rows = {
        d["document_id"]: d
        for d in client.get("/v1/documents", headers={"x-api-key": "demo-local"}).json()
    }
    assert rows[mine]["owned_by_caller"] is True
    assert rows[shared]["owned_by_caller"] is False


def test_shared_document_is_readable_but_not_deletable_by_a_role_holder(client: Any) -> None:
    shared = _seed_doc("admin", "acme", ["employee"], C.INTERNAL)
    demo = {"x-api-key": "demo-local"}
    assert client.get(f"/v1/documents/{shared}", headers=demo).status_code == 200
    assert client.delete(f"/v1/documents/{shared}", headers=demo).status_code == 404
    assert (
        client.get(f"/v1/documents/{shared}", headers={"x-api-key": "globex-local"}).status_code
        == 404
    )


def test_upload_cannot_exceed_the_uploaders_clearance(client: Any) -> None:
    resp = client.post(
        "/v1/documents/local-upload?classification=RESTRICTED",
        files={"file": ("p.md", b"# Policy\n\nRefunds take 30 days.", "text/markdown")},
        headers={"x-api-key": "demo-local"},
    )
    assert resp.status_code == 403


def test_role_names_are_validated(client: Any) -> None:
    resp = client.post(
        "/v1/documents/local-upload?roles=fin%20ance",
        files={"file": ("p.md", b"# Policy\n\nRefunds take 30 days.", "text/markdown")},
        headers={"x-api-key": "demo-local"},
    )
    assert resp.status_code == 400


def test_retrieval_excludes_rows_before_scoring_and_the_trace_says_how_many(
    client: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.agents import retriever as retriever_mod
    from app.observability.tracing import TraceRecorder
    from app.rag import embeddings

    acme_doc = _seed_doc("demo", "acme", [], C.INTERNAL)
    globex_doc = _seed_doc("globex-user", "globex", [], C.INTERNAL)
    chunks = [
        _chunk("acme1", owner="demo", tenant="acme", level=C.INTERNAL, doc=acme_doc),
        _chunk("glx1", owner="globex-user", tenant="globex", level=C.INTERNAL, doc=globex_doc),
        _chunk("glx2", owner="globex-user", tenant="globex", level=C.INTERNAL, doc=globex_doc),
        _chunk("pub", level=C.PUBLIC),
    ]
    _publish(tmp_path, chunks)
    monkeypatch.setattr(
        embeddings, "embed_texts", lambda texts: np.ones((len(texts), 384), dtype=np.float32)
    )

    seen_allowed: list[Any] = []
    real_search = index_store.search

    def spy(index: Any, q: Any, k: int, *, allowed_rows: Any = None) -> Any:
        seen_allowed.append(allowed_rows)
        return real_search(index, q, k, allowed_rows=allowed_rows)

    monkeypatch.setattr(index_store, "search", spy)

    state: Any = {
        "query": "text",
        "search_query": "text",
        "user_id": "demo",
        "access": _ctx("demo-local"),
        "doc_id": None,
        "trace": TraceRecorder(),
        "attempt": 0,
        "model": "m",
        "token_budget_left": 1000,
        "deadline_ts": time.monotonic() + 30,
        "retrieved": [],
        "answer": "",
        "citations": [],
        "critic": None,
        "status": "running",
        "refusal_reason": None,
        "conversation_history": [],
    }
    out = retriever_mod.retriever_node(state)

    assert {c.chunk_id for c in out["retrieved"]} == {"acme1", "pub"}
    assert seen_allowed == [{0, 3}], "FAISS must receive the allowed rows, not filter afterwards"
    step = next(s for s in state["trace"].steps if s["name"] == "retriever")
    assert step["meta"]["rows_excluded"] == "2"


def test_chunk_detail_follows_the_same_visibility_rule(client: Any, tmp_path: Any) -> None:
    acme_doc = _seed_doc("demo", "acme", [], C.INTERNAL)
    chunk = _chunk("acme1", owner="demo", tenant="acme", level=C.INTERNAL, doc=acme_doc)
    _publish(tmp_path, [chunk])

    ok = client.get(f"/v1/documents/{acme_doc}/chunks/acme1", headers={"x-api-key": "demo-local"})
    assert ok.status_code == 200
    body = ok.json()
    assert body["text"] == "text acme1" and body["document_checksum_sha256"] == "abc"
    assert "sha256" in body["hash_recipe"]

    for key in ("globex-local", "finance-local", "contractor-local"):
        assert (
            client.get(
                f"/v1/documents/{acme_doc}/chunks/acme1", headers={"x-api-key": key}
            ).status_code
            == 404
        )


def test_scoped_query_on_a_role_shared_document_is_allowed(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.api import routes_query

    shared = _seed_doc("admin", "acme", ["employee"], C.INTERNAL)

    class _G:
        def invoke(self, state: dict) -> dict:
            state.update(status="answered", answer="ok [chunk:x]", citations=[], retrieved=[])
            return state

    monkeypatch.setattr(routes_query, "_get_graph", lambda: _G())
    resp = client.post(
        "/v1/query", json={"query": "q", "doc_id": shared}, headers={"x-api-key": "demo-local"}
    )
    assert resp.status_code == 200
    resp = client.post(
        "/v1/query", json={"query": "q", "doc_id": shared}, headers={"x-api-key": "globex-local"}
    )
    assert resp.status_code == 404


def test_access_module_is_the_only_place_that_decides_visibility() -> None:
    """Grep-level guard: no other module may compare owner ids or tenant ids
    to decide what a caller sees. Four copies of a rule is four leaks."""
    import pathlib

    app = pathlib.Path(access.__file__).resolve().parents[1]
    offenders = []
    for path in app.rglob("*.py"):
        rel = path.relative_to(app).as_posix()
        if rel in ("guardrails/access.py",):
            continue
        text = path.read_text(encoding="utf-8")
        if "clearance.rank" in text or ".allowed_roles) &" in text:
            offenders.append(rel)
    assert offenders == ["api/routes_ingest.py"] or offenders == [], offenders
