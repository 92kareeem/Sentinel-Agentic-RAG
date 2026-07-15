"""Grounding verifier tests — pure string logic, fully offline."""

from app.guardrails import grounding
from app.models.schemas import Chunk


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id="d", section_path="S", text=text,
        token_count=10, char_start=0, char_end=len(text),
    )


CHUNKS = [
    _chunk("c1", "Refunds take 3 days in the US with a 5% restocking fee."),
    _chunk("c2", "Engineers get a $2,400 laptop budget every 3 years."),
]


def test_good_answer_passes() -> None:
    answer = "Refunds take 3 days [chunk:c1]. The budget is 2,400 dollars [chunk:c2]."
    r = grounding.verify(answer, CHUNKS)
    assert r.ok and r.stripped_ratio == 0.0
    assert set(r.valid_chunk_ids) == {"c1", "c2"}


def test_fabricated_number_is_stripped() -> None:
    answer = (
        "Refunds take 3 days [chunk:c1]. Refunds take 99 days [chunk:c1]. Fee is 5% [chunk:c1]."
    )
    r = grounding.verify(answer, CHUNKS)
    assert "99" not in r.clean_answer
    assert not r.ok  # 1/3 stripped > 30%


def test_unknown_chunk_id_is_stripped() -> None:
    answer = "Something made up [chunk:ghost]."
    r = grounding.verify(answer, CHUNKS)
    assert not r.ok and r.stripped_ratio == 1.0


def test_insufficient_context_passes_through() -> None:
    r = grounding.verify("INSUFFICIENT_CONTEXT", CHUNKS)
    assert r.ok and r.clean_answer == "INSUFFICIENT_CONTEXT"


def test_uncited_answer_fails_closed() -> None:
    r = grounding.verify("Trust me, refunds are instant.", CHUNKS)
    assert not r.ok
