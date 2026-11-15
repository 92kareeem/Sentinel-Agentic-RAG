"""Deterministic grounding verifier — the last gate before the user.

The engine lives in the standalone ``sentinel-proof`` package
(``packages/sentinel-proof``), which anyone can use with any RAG stack. This
module is the platform's adapter: it maps our ``Chunk`` records onto the
package's ``Passage`` and the package's ``Verdict`` onto the ``GroundingResult``
the graph, the demo and the tests already consume.

Why the gate exists, in one paragraph: the critic is an LLM and can be wrong;
this check is pure string logic and cannot hallucinate. Per sentence, it asks
whether a retrieved passage actually supports the claim and whether the
citation points at it. Supported sentences ship; sentences another passage
supports are re-cited; the rest are stripped. If more than half the answer is
stripped the answer is untrustworthy and goes back to the repair loop. See the
package for the full method and its tests.
"""

from dataclasses import dataclass

from sentinel_proof import MAX_STRIPPED_RATIO, MIN_LEXICAL_OVERLAP, Passage, normalize_citations
from sentinel_proof import verify as _verify

from app.models.schemas import Chunk

__all__ = [
    "MAX_STRIPPED_RATIO",
    "MIN_LEXICAL_OVERLAP",
    "GroundingResult",
    "normalize_citations",
    "verify",
]


@dataclass(frozen=True)
class GroundingResult:
    clean_answer: str
    ok: bool
    stripped_ratio: float
    valid_chunk_ids: list[str]
    # Sentences whose citation pointed at a chunk that does not support them,
    # where another retrieved chunk does. A steady rate here means the
    # synthesizer is attributing badly, which is worth knowing even though the
    # user now sees a corrected citation.
    repaired_citations: int = 0


def _passage(chunk: Chunk) -> Passage:
    return Passage(
        id=chunk.chunk_id,
        text=chunk.text,
        section=chunk.section_path,
        source=chunk.source_filename,
        page=chunk.page_number,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        document_sha256=chunk.document_checksum,
        provenance_hash=chunk.provenance_hash,
    )


def verify(answer: str, retrieved: list[Chunk], question: str = "") -> GroundingResult:
    v = _verify(answer, [_passage(c) for c in retrieved], question)
    if v.is_refusal:
        # The synthesizer's own refusal sentinel passes through unchanged; the
        # graph routes on it and the API turns it into a RefusalResponse.
        return GroundingResult(normalize_citations(answer), ok=True, stripped_ratio=0.0,
                               valid_chunk_ids=[])
    return GroundingResult(
        clean_answer=v.answer,
        ok=v.ok,
        stripped_ratio=v.stripped_ratio,
        valid_chunk_ids=list(v.cited_ids),
        repaired_citations=v.repaired_citations,
    )
