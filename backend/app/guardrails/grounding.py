"""Deterministic grounding verifier — the last gate before the user.

Role in architecture: the critic is an LLM and can be wrong; this check is
pure string logic and cannot hallucinate. Every cited chunk_id must exist,
and the numbers in each sentence must literally appear in the cited chunk.
Failing sentences are stripped; if >30% of sentences fail, the whole answer
is untrustworthy and is treated like a critic failure (repair or refuse).
"""

import re
from dataclasses import dataclass

from app.models.schemas import Chunk

_CITED_SENT_RE = re.compile(r"([^.!?\n]+?)\s*\[chunk:([\w-]+)\]\s*[.!?]?", re.S)
_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")

MAX_STRIPPED_RATIO = 0.30


@dataclass(frozen=True)
class GroundingResult:
    clean_answer: str
    ok: bool
    stripped_ratio: float
    valid_chunk_ids: list[str]


def _numbers_in(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER_RE.findall(text)}


def verify(answer: str, retrieved: list[Chunk]) -> GroundingResult:
    if answer.strip() == "INSUFFICIENT_CONTEXT":
        return GroundingResult(answer, ok=True, stripped_ratio=0.0, valid_chunk_ids=[])

    by_id = {c.chunk_id: c for c in retrieved}
    sentences = _CITED_SENT_RE.findall(answer)
    if not sentences:  # no parseable cited sentences at all -> fail closed
        return GroundingResult(answer, ok=False, stripped_ratio=1.0, valid_chunk_ids=[])

    kept: list[str] = []
    valid_ids: list[str] = []
    stripped = 0
    for sentence, chunk_id in sentences:
        chunk = by_id.get(chunk_id)
        if chunk is None:
            stripped += 1
            continue
        chunk_numbers = _numbers_in(chunk.text)
        if not _numbers_in(sentence) <= chunk_numbers:
            stripped += 1
            continue
        kept.append(f"{sentence.strip()} [chunk:{chunk_id}]")
        valid_ids.append(chunk_id)

    ratio = stripped / len(sentences)
    return GroundingResult(
        clean_answer=". ".join(kept) + ("." if kept else ""),
        ok=ratio <= MAX_STRIPPED_RATIO,
        stripped_ratio=ratio,
        valid_chunk_ids=valid_ids,
    )
