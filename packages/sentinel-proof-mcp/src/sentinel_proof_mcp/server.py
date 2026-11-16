"""The sentinel-proof MCP server: one tool and one prompt.

``verify_answer`` takes the passages an agent retrieved and the answer it
drafted, and returns what the deterministic grounding gate decided, claim by
claim, plus the text that is safe to show. Nothing here calls a model or the
network; the verdict is computed by :mod:`sentinel_proof` in-process.
"""

from __future__ import annotations

from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from sentinel_proof import (
    MAX_STRIPPED_RATIO,
    Passage,
    Verdict,
    passages_from_texts,
    verify,
)
from sentinel_proof import (
    __version__ as proof_version,
)
from typing_extensions import TypedDict  # pydantic needs this one on Python < 3.12

from . import __version__

SERVER_NAME = "sentinel-proof"

INSTRUCTIONS = """\
sentinel-proof checks a drafted answer against the passages it was written from.
Call `verify_answer` after retrieving sources and before showing the user an
answer that states facts from them. The answer must cite passages as [1], [2]
(1-based positions in the `passages` list) or as [chunk:<id>] when passages
carry ids; the `cite_for_verification` prompt gives the exact instruction to
follow when drafting. Show the user `safe_text`, never the unverified draft:
unsupported sentences have been removed from it, wrong citations re-pointed,
and an answer that is mostly unsupported has been replaced by a refusal.
The check is deterministic (word overlap and numbers), not semantic.
"""


class PassageInput(TypedDict, total=False):
    """A retrieved passage. Only ``text`` is required."""

    text: str
    id: str
    source: str
    section: str
    page: int


class ClaimOutput(TypedDict):
    text: str
    status: Literal["supported", "repaired", "stripped"]
    cited: list[str]
    supported_by: list[str]
    unsupported_numbers: list[str]


class VerdictOutput(TypedDict):
    ok: bool
    passed: bool
    is_refusal: bool
    refusal: str | None
    summary: str
    safe_text: str
    answer: str
    claims: list[ClaimOutput]
    stripped_ratio: float
    repaired_citations: int
    cited_ids: list[str]


def _to_passages(raw: list[str | PassageInput]) -> list[Passage]:
    if not raw:
        raise ToolError("passages must not be empty: pass the texts the answer was written from")
    if all(isinstance(p, str) for p in raw):
        return passages_from_texts([p for p in raw if isinstance(p, str)])
    out: list[Passage] = []
    for i, p in enumerate(raw, 1):
        if isinstance(p, str):
            out.append(Passage(text=p))
            continue
        text = p.get("text", "")
        if not isinstance(text, str) or not text.strip():
            raise ToolError(f"passage {i} has no text")
        out.append(
            Passage(
                text=text,
                id=str(p.get("id", "") or ""),
                source=str(p.get("source", "") or ""),
                section=str(p.get("section", "") or ""),
                page=p.get("page"),
            )
        )
    return out


def summarize(v: Verdict, citation_style: str, max_stripped_ratio: float) -> str:
    """One line the model can act on without reading the whole verdict."""
    if v.is_refusal:
        return (
            "The draft is the model's own refusal (INSUFFICIENT_CONTEXT); there is nothing to "
            "verify. Tell the user the sources do not answer the question."
        )
    total = len(v.claims)
    if total == 0:
        how = "[1], [2]" if citation_style == "numeric" else "[chunk:<id>]"
        return (
            "No citations could be parsed, so nothing could be checked and the gate fails "
            f"closed. Rewrite the answer with a citation after every sentence, as {how}, "
            "then call verify_answer again."
        )
    kept = sum(1 for c in v.claims if c.status.value != "stripped")
    stripped = total - kept
    if v.ok:
        parts = [f"Verified: {kept} of {total} claims supported"]
        if v.repaired_citations:
            parts.append(f"{v.repaired_citations} citation(s) re-pointed to the right passage")
        if stripped:
            parts.append(f"{stripped} unsupported claim(s) removed")
        return ". ".join(parts) + ". Show the user safe_text."
    figures = sorted({n for c in v.claims for n in c.unsupported_numbers})
    why = f" Figures no passage contains: {', '.join(figures)}." if figures else ""
    return (
        f"Refused: {stripped} of {total} claims are not supported by the passages "
        f"({v.stripped_ratio:.0%}, limit {max_stripped_ratio:.0%}).{why} Do not show the "
        "draft. Either answer only from what the passages say, retrieve more, or tell the "
        "user the sources do not cover it (safe_text is a suitable refusal)."
    )


def verdict_output(v: Verdict, citation_style: str, max_stripped_ratio: float) -> VerdictOutput:
    d = v.to_dict()
    return {
        "ok": bool(d["ok"]),
        "passed": bool(d["passed"]),
        "is_refusal": bool(d["is_refusal"]),
        "refusal": d["refusal"],
        "summary": summarize(v, citation_style, max_stripped_ratio),
        "safe_text": str(d["safe_text"]),
        "answer": str(d["answer"]),
        "claims": [
            {
                "text": c["text"],
                "status": c["status"],
                "cited": list(c["cited"]),
                "supported_by": list(c["supported_by"]),
                "unsupported_numbers": list(c["unsupported_numbers"]),
            }
            for c in d["claims"]
        ],
        "stripped_ratio": float(d["stripped_ratio"]),
        "repaired_citations": int(d["repaired_citations"]),
        "cited_ids": list(d["cited_ids"]),
    }


def build_server() -> MCPServer:
    server = MCPServer(
        SERVER_NAME,
        title="sentinel-proof",
        instructions=INSTRUCTIONS,
        website_url="https://github.com/92kareeem/sentinel-rag/tree/main/packages/sentinel-proof-mcp",
        version=__version__,
    )

    @server.tool(
        title="Verify an answer against its sources",
        annotations=ToolAnnotations(
            read_only_hint=True, idempotent_hint=True, open_world_hint=False
        ),
    )
    def verify_answer(
        answer: str,
        passages: list[str | PassageInput],
        question: str = "",
        citation_style: Literal["numeric", "tag"] = "numeric",
        max_stripped_ratio: float = MAX_STRIPPED_RATIO,
    ) -> VerdictOutput:
        """Check a drafted answer against the passages it cites, sentence by sentence.

        Every number in a sentence must appear in a passage it cites, and the
        sentence's words must overlap that passage. A sentence whose citation is
        wrong is re-pointed to the passage that does support it ("repaired"); a
        sentence no passage supports is removed ("stripped"); if more than
        max_stripped_ratio of the sentences are stripped the whole answer is
        refused and `safe_text` is a refusal instead.

        `passages` are the retrieved texts in the order they were shown to the
        model, as strings or as {"text", "id", "source", "section", "page"}.
        With citation_style "numeric" the answer cites them as [1], [2, 3]
        (1-based positions); with "tag" it cites [chunk:<id>]. `question` lets
        figures the user typed pass without being counted as fabrications.
        Deterministic: no model, no network. Returns the verdict with
        per-claim detail, a one-line `summary`, and `safe_text` to show.
        """
        if not answer.strip():
            raise ToolError("answer must not be empty")
        if not 0.0 <= max_stripped_ratio <= 1.0:
            raise ToolError("max_stripped_ratio must be between 0 and 1")
        try:
            v = verify(
                answer,
                _to_passages(passages),
                question or "",
                citation_style=citation_style,
                max_stripped_ratio=max_stripped_ratio,
            )
        except ValueError as exc:  # e.g. two passages with the same id
            raise ToolError(str(exc)) from exc
        return verdict_output(v, citation_style, max_stripped_ratio)

    @server.prompt(title="Draft an answer that verify_answer can check")
    def cite_for_verification(citation_style: Literal["numeric", "tag"] = "numeric") -> str:
        """Instructions to give the model before it drafts, so every sentence is citable."""
        how = (
            "Cite each sentence with the 1-based position of the passage it comes from, "
            "in square brackets: [1], or [1, 3] when a sentence rests on two passages."
            if citation_style == "numeric"
            else "Cite each sentence with the passage's id in the form [chunk:<id>], "
            "for example [chunk:p2]; use several tags when a sentence rests on several."
        )
        return (
            "Answer only from the passages provided. Put a citation at the end of every "
            f"sentence that states a fact. {how} Copy numbers, dates, amounts and names "
            "exactly as the passage gives them; do not compute, convert or round. If the "
            "passages do not answer the question, reply with exactly INSUFFICIENT_CONTEXT "
            "and nothing else. Then call verify_answer with the draft and the passages, "
            "and show the user its safe_text."
        )

    @server.resource(
        "sentinel-proof://about",
        name="about",
        title="About this server",
        description="Versions and the verification rules in one place.",
        mime_type="text/plain",
    )
    def about() -> str:
        return (
            f"sentinel-proof-mcp {__version__} over sentinel-proof {proof_version}.\n"
            "Rules, per cited sentence: every number must appear in a cited passage; content "
            "words must overlap the passage by at least 25%. Outcomes: supported, repaired "
            "(citation re-pointed), stripped (removed). Refused when more than 50% of the "
            "claims are stripped (max_stripped_ratio). No citations at all fails closed.\n"
            "Docs: https://github.com/92kareeem/sentinel-rag/tree/main/packages/sentinel-proof-mcp\n"
            "Playground: https://92kareeem.github.io/sentinel-rag/\n"
        )

    return server
