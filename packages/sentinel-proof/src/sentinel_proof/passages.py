"""The evidence an answer is checked against.

A :class:`Passage` is one retrieved piece of text. Only ``text`` is required;
the rest lets the gate do more: ``section`` is counted as evidence (a heading
like "Item 7.2 > Refunds" grounds both the topic and the figure 7.2), and the
span plus document checksum let anyone recompute the provenance hash.

Adapters at the bottom build passages from the objects other libraries hand
you. They are duck-typed on purpose, so this package imports nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .provenance import provenance_hash


@dataclass(frozen=True)
class Passage:
    text: str
    id: str = ""
    section: str = ""
    source: str = ""  # filename, URL, title: whatever a reader would recognise
    page: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    document_sha256: str = ""  # checksum of the file the span was cut from
    provenance_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @property
    def evidence(self) -> str:
        """What a claim may be supported by: the section path plus the text."""
        return f"{self.section}\n{self.text}" if self.section else self.text

    def with_provenance(self) -> Passage:
        """A copy whose ``provenance_hash`` is computed from the span.

        Needs ``document_sha256``, ``char_start`` and ``char_end``; without
        them there is nothing to prove and the passage is returned unchanged.
        """
        if not self.document_sha256 or self.char_start is None or self.char_end is None:
            return self
        return Passage(
            **{
                **self.__dict__,
                "provenance_hash": provenance_hash(
                    self.document_sha256, self.char_start, self.char_end, self.text
                ),
            }
        )


def _with_ids(passages: Iterable[Passage]) -> list[Passage]:
    """Give every passage without an id a positional one (``p1``, ``p2``...).

    Ids must be unique: two passages with the same id would make a citation
    ambiguous, and the gate would silently check against the wrong one.
    """
    out: list[Passage] = []
    seen: set[str] = set()
    for i, p in enumerate(passages, start=1):
        pid = p.id or f"p{i}"
        if pid in seen:
            raise ValueError(f"duplicate passage id {pid!r}")
        seen.add(pid)
        out.append(p if p.id else Passage(**{**p.__dict__, "id": pid}))
    return out


def passages_from_texts(texts: Sequence[str], ids: Sequence[str] | None = None) -> list[Passage]:
    """Passages from plain strings, for the simplest possible call site."""
    if ids is not None and len(ids) != len(texts):
        raise ValueError("ids and texts must have the same length")
    return _with_ids(Passage(text=t, id=ids[i] if ids else "") for i, t in enumerate(texts))


# ----------------------------------------------------------------- adapters


def _meta(obj: Any) -> dict[str, Any]:
    md = getattr(obj, "metadata", None)
    return dict(md) if isinstance(md, dict) else {}


def from_langchain(documents: Iterable[Any]) -> list[Passage]:
    """Passages from LangChain ``Document`` objects (``page_content`` + ``metadata``).

    Recognised metadata keys, all optional: ``id``/``chunk_id``, ``source``,
    ``page``, ``section``, ``start_index`` (what ``add_start_index=True`` on the
    text splitters writes), ``char_start``/``char_end``, ``document_sha256``.
    """
    out: list[Passage] = []
    for d in documents:
        md = _meta(d)
        text = getattr(d, "page_content", None)
        if text is None:
            raise TypeError("expected an object with .page_content (a LangChain Document)")
        start = md.get("char_start", md.get("start_index"))
        end = md.get("char_end")
        if start is not None and end is None:
            end = int(start) + len(text)
        out.append(
            Passage(
                text=text,
                id=str(md.get("id") or md.get("chunk_id") or getattr(d, "id", "") or ""),
                section=str(md.get("section") or ""),
                source=str(md.get("source") or md.get("title") or ""),
                page=md.get("page"),
                char_start=start,
                char_end=end,
                document_sha256=str(md.get("document_sha256") or ""),
                metadata=md,
            )
        )
    return _with_ids(out)


def from_llamaindex(nodes: Iterable[Any]) -> list[Passage]:
    """Passages from LlamaIndex nodes, or the ``NodeWithScore`` wrappers a
    retriever returns. Uses ``node_id``, ``get_content()``, ``start_char_idx``,
    ``end_char_idx`` and the ``file_name``/``page_label`` metadata keys."""
    out: list[Passage] = []
    for n in nodes:
        node = getattr(n, "node", n)
        get_content = getattr(node, "get_content", None)
        text = get_content() if callable(get_content) else getattr(node, "text", None)
        if text is None:
            raise TypeError("expected a LlamaIndex node or NodeWithScore")
        md = _meta(node)
        page = md.get("page_label")
        out.append(
            Passage(
                text=text,
                id=str(getattr(node, "node_id", "") or getattr(node, "id_", "") or ""),
                section=str(md.get("section") or ""),
                source=str(md.get("file_name") or md.get("source") or ""),
                page=int(page) if isinstance(page, (int, str)) and str(page).isdigit() else None,
                char_start=getattr(node, "start_char_idx", None),
                char_end=getattr(node, "end_char_idx", None),
                document_sha256=str(md.get("document_sha256") or ""),
                metadata=md,
            )
        )
    return _with_ids(out)
