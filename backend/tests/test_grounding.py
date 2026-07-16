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


def test_fabricated_number_is_stripped_but_answer_ships() -> None:
    # one fabricated sentence of three: stripped from the answer, rest still ships
    answer = (
        "Refunds take 3 days [chunk:c1]. Refunds take 99 days [chunk:c1]. Fee is 5% [chunk:c1]."
    )
    r = grounding.verify(answer, CHUNKS)
    assert "99" not in r.clean_answer  # the hallucinated sentence is gone
    assert r.ok  # 1/3 stripped <= 50%, so the cleaned answer is served, not refused
    assert "3 days" in r.clean_answer and "5%" in r.clean_answer


def test_mostly_fabricated_answer_refuses() -> None:
    # two of three sentences invent numbers absent from all context -> refuse
    answer = (
        "Refunds take 88 days [chunk:c1]. Fee is 77% [chunk:c1]. Refunds take 3 days [chunk:c1]."
    )
    r = grounding.verify(answer, CHUNKS)
    assert not r.ok and r.stripped_ratio > 0.5


def test_number_grounded_in_other_retrieved_chunk_passes() -> None:
    # sentence cites c1 but its number lives in c2 — still grounded in context
    answer = "The laptop budget is 2,400 dollars [chunk:c1]."
    r = grounding.verify(answer, CHUNKS)
    assert r.ok and "2,400" in r.clean_answer


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
