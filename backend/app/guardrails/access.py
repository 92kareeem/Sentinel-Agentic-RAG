"""Document and chunk visibility: one rule, applied everywhere.

Role in architecture: the registry, the retriever, the chunk endpoint and
the document list all ask the same question — may this caller see this
document? — and they all ask it here. A rule that lives in four places is
four chances to disagree, and in an access-control rule a disagreement is
a leak.

The rule, in order:

  1. tenant    the document belongs to the caller's tenant, or to no tenant
               (the seeded demo corpus, shared with everyone)
  2. who       the caller owns it, or is a tenant admin, or holds one of the
               document's allowed roles; an empty role list means owner-only
  3. clearance the caller's clearance is at least the document's level
  4. scope     when a query is scoped to one document, exact id match

Enforced BEFORE scoring in retrieval (rag/index_store.search takes the
allowed rows as a FAISS selector; bm25 masks them), so an unauthorized chunk
is never ranked, never loaded into agent state, never in a prompt.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from app.models.schemas import AccessContext, ClassificationLevel


class _Protected(Protocol):
    """What a document and a chunk have in common for this decision."""

    owner_id: str
    tenant_id: str
    allowed_roles: list[str]
    classification: ClassificationLevel


def context_for(user: dict[str, Any]) -> AccessContext:
    """Build the caller's context from a resolved user record."""
    clearance = user.get("clearance") or ClassificationLevel.INTERNAL
    return AccessContext(
        user_id=str(user["user_id"]),
        tenant_id=str(user.get("tenant_id") or ""),
        roles=[str(r) for r in (user.get("roles") or [])],
        clearance=ClassificationLevel(clearance),
        is_admin=bool(user.get("is_admin")),
    )


def can_see(ctx: AccessContext, item: _Protected) -> bool:
    shared_corpus = item.tenant_id == "" and item.owner_id == ""
    if not shared_corpus and item.tenant_id and item.tenant_id != ctx.tenant_id:
        return False
    if item.tenant_id == "" and item.owner_id and item.owner_id != ctx.user_id:
        # A document uploaded before tenants existed: owner-only, as before.
        return False
    if not shared_corpus and item.tenant_id:
        owner = item.owner_id == ctx.user_id
        admin = ctx.is_admin
        role = bool(set(item.allowed_roles) & set(ctx.roles))
        if not (owner or admin or role):
            return False
    return ctx.clearance.rank >= item.classification.rank


def document_visible(ctx: AccessContext, record: _Protected | None) -> bool:
    return record is not None and can_see(ctx, record)


def visible_rows(
    chunks: Iterable[_Protected], ctx: AccessContext, doc_id: str | None
) -> set[int] | None:
    """Row ids this caller may have scored.

    Returns None only when nothing needs excluding — an admin over a
    single-tenant corpus with no scope — so the common path stays cheap.
    Document scope matches on EXACT id: a `startswith` test once widened a
    scope to every document whose id shared a prefix.
    """
    rows: set[int] = set()
    excluded = False
    for i, chunk in enumerate(chunks):
        if doc_id is not None and getattr(chunk, "doc_id", None) != doc_id:
            excluded = True
            continue
        if not can_see(ctx, chunk):
            excluded = True
            continue
        rows.add(i)
    return rows if excluded else None
