"""Adapters are duck-typed; provenance hashes match the published recipe."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import pytest

from sentinel_proof import (
    HASH_RECIPE,
    Passage,
    check_provenance,
    document_sha256,
    from_langchain,
    from_llamaindex,
    passages_from_texts,
    provenance_hash,
    verify,
)


@dataclass
class FakeDocument:  # the shape of langchain_core.documents.Document
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass
class FakeNode:  # the shape of llama_index.core.schema.TextNode
    text: str
    node_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    start_char_idx: int | None = None
    end_char_idx: int | None = None

    def get_content(self) -> str:
        return self.text


@dataclass
class FakeNodeWithScore:
    node: FakeNode
    score: float = 0.5


def test_langchain_documents_become_passages_and_verify() -> None:
    docs = [
        FakeDocument(
            "Customers may request a refund within 30 days.",
            {"source": "policy.md", "start_index": 120, "chunk_id": "pol-1"},
        ),
        FakeDocument("Unrelated text about parking.", {"source": "facilities.md"}),
    ]
    passages = from_langchain(docs)
    assert [p.id for p in passages] == ["pol-1", "p2"]
    assert passages[0].source == "policy.md"
    assert (passages[0].char_start, passages[0].char_end) == (120, 120 + len(docs[0].page_content))
    assert verify(
        "Refunds can be requested within 30 days [1].", passages, citation_style="numeric"
    ).ok


def test_llamaindex_nodes_and_wrappers_become_passages() -> None:
    node = FakeNode(
        "Refund requests above 500 USD require manager approval.",
        "n-7",
        {"file_name": "approvals.pdf", "page_label": "3"},
        10,
        65,
    )
    passages = from_llamaindex([FakeNodeWithScore(node), FakeNode("Other.", "n-8")])
    assert passages[0].id == "n-7" and passages[0].source == "approvals.pdf"
    assert passages[0].page == 3 and (passages[0].char_start, passages[0].char_end) == (10, 65)
    assert passages[1].id == "n-8"


def test_wrong_types_fail_loudly() -> None:
    with pytest.raises(TypeError):
        from_langchain(["a plain string"])
    with pytest.raises(TypeError):
        from_llamaindex([object()])


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        passages_from_texts(["a", "b"], ids=["x", "x"])


def test_hash_matches_the_published_recipe() -> None:
    """A client in any language must be able to verify without reading this code."""
    h = provenance_hash("abc", 120, 410, "the quoted text")
    assert h == hashlib.sha256(b"abc|120|410|the quoted text").hexdigest()
    assert "sha256" in HASH_RECIPE and "'|'" in HASH_RECIPE


def test_hash_changes_when_any_input_changes() -> None:
    base = provenance_hash("abc", 1, 2, "t")
    assert base != provenance_hash("abd", 1, 2, "t")
    assert base != provenance_hash("abc", 1, 3, "t")
    assert base != provenance_hash("abc", 1, 2, "u")


def test_with_provenance_fills_the_hash_only_when_it_can() -> None:
    data = b"# Policy\nRefunds take 30 days."
    text = "Refunds take 30 days."
    start = data.decode().index(text)
    p = Passage(
        text=text,
        document_sha256=document_sha256(data),
        char_start=start,
        char_end=start + len(text),
    ).with_provenance()
    assert len(p.provenance_hash) == 64
    assert check_provenance(
        document_sha256=p.document_sha256,
        char_start=start,
        char_end=start + len(text),
        text=text,
        expected=p.provenance_hash,
    )
    assert Passage(text="no span").with_provenance().provenance_hash == ""
