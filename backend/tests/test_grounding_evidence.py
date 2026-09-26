"""The grounding gate's two jobs: is the claim supported, and by the passage
it names?

Both failure modes below produce answers that look correct. A wrong number
reads as authoritative; a wrong citation only breaks when someone clicks it.
For a product whose whole promise is "every claim is supported by your
documents", those are the failures that matter.
"""

from __future__ import annotations

from app.guardrails import grounding
from app.models.schemas import Chunk


def _chunk(cid: str, text: str, section: str = "Policy") -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id="doc-1",
        section_path=section,
        text=text,
        token_count=len(text.split()),
        char_start=0,
        char_end=len(text),
    )


REFUNDS = _chunk(
    "refunds",
    "Customers may request a refund within 30 days of purchase.",
    section="Refund policy",
)
HEADCOUNT = _chunk(
    "headcount",
    "The support team grew to 30 employees across two regions.",
    section="Team",
)
APPROVALS = _chunk(
    "approvals",
    "Refund requests above 500 USD require manager approval before processing.",
    section="Refund approvals",
)


# ------------------------------------------- numbers must live in the evidence


def test_a_number_from_an_unrelated_passage_no_longer_grounds_a_claim() -> None:
    """THE regression test.

    Numbers were checked against the union of every retrieved chunk, so a
    fabricated refund window passed whenever any retrieved passage happened to
    contain the same digits — here a headcount. For a policy assistant, a
    plausible-but-wrong number is the most damaging output there is, and this
    check could not catch one.
    """
    answer = "Customers may request a refund within 60 days of purchase. [chunk:refunds]"
    result = grounding.verify(answer, [REFUNDS, HEADCOUNT])

    assert "60 days" not in result.clean_answer


def test_the_same_claim_with_the_right_number_still_passes() -> None:
    answer = "Customers may request a refund within 30 days of purchase. [chunk:refunds]"
    result = grounding.verify(answer, [REFUNDS, HEADCOUNT])

    assert result.ok
    assert "30 days" in result.clean_answer


def test_a_number_echoed_from_the_question_is_not_a_fabrication() -> None:
    """Guards against the obvious over-correction. "Is 20 days within the
    window?" invites an answer that restates 20 while the evidence contains
    only 30 — stripping that would refuse a correct answer."""
    answer = "At 20 days the request is inside the 30 day refund window. [chunk:refunds]"
    result = grounding.verify(
        answer, [REFUNDS], question="Can we refund at 20 days after purchase?"
    )

    assert result.ok
    assert "20 days" in result.clean_answer


def test_a_list_marker_is_not_read_as_a_factual_claim() -> None:
    """An enumerated answer starts "1." — that digit is presentation, not a
    claim about the number one, and reading it as one strips a good sentence."""
    answer = "1. Customers may request a refund within 30 days. [chunk:refunds]"
    result = grounding.verify(answer, [REFUNDS])

    assert result.ok
    assert "30 days" in result.clean_answer


def test_a_figure_that_lives_only_in_the_section_heading_still_grounds() -> None:
    chunk = _chunk("s", "Requests must be filed in writing.", section="Item 7.2 > Refunds")
    answer = "Requests must be filed in writing under Item 7.2. [chunk:s]"

    assert grounding.verify(answer, [chunk]).ok


def test_thousands_separators_do_not_look_like_different_numbers() -> None:
    chunk = _chunk("c", "The annual cap is 1,200 USD per customer.", section="Caps")
    answer = "The annual cap is 1200 USD per customer. [chunk:c]"

    assert grounding.verify(answer, [chunk]).ok


# ----------------------------------------------- citations must point at truth


def test_a_citation_naming_the_wrong_passage_is_repaired_not_accepted() -> None:
    """THE second regression test.

    A claim supported by some other retrieved chunk used to be accepted with
    its wrong citation still attached. The answer body looks right and breaks
    the moment anyone clicks through — and a user who checks one citation and
    finds it wrong stops trusting every other one.
    """
    answer = "Refunds above 500 USD require manager approval. [chunk:refunds]"
    result = grounding.verify(answer, [REFUNDS, APPROVALS])

    assert result.ok
    assert "[chunk:approvals]" in result.clean_answer
    assert "[chunk:refunds]" not in result.clean_answer
    assert result.repaired_citations == 1
    assert result.valid_chunk_ids == ["approvals"]


def test_a_correct_citation_is_left_alone_and_not_counted_as_repaired() -> None:
    answer = "Refunds above 500 USD require manager approval. [chunk:approvals]"
    result = grounding.verify(answer, [REFUNDS, APPROVALS])

    assert result.repaired_citations == 0
    assert "[chunk:approvals]" in result.clean_answer


def test_a_claim_nothing_retrieved_supports_is_still_stripped() -> None:
    """Repair must not become a way to attach any citation to anything."""
    answer = "Enterprise customers receive a dedicated success manager. [chunk:refunds]"
    result = grounding.verify(answer, [REFUNDS, HEADCOUNT])

    assert "success manager" not in result.clean_answer


def test_a_fabricated_chunk_id_can_be_repaired_when_the_claim_is_real() -> None:
    """A tag naming a chunk that was never retrieved is a model slip, not
    necessarily a fabricated claim. Finding the real source beats refusing."""
    answer = "Customers may request a refund within 30 days. [chunk:does-not-exist]"
    result = grounding.verify(answer, [REFUNDS])

    assert result.ok
    assert "[chunk:refunds]" in result.clean_answer


def test_repair_picks_the_passage_that_says_the_most_about_the_claim() -> None:
    """Several chunks can mention a topic. The citation has to land on the one
    a reader will recognise as the source."""
    vague = _chunk("vague", "Refunds are covered elsewhere in this handbook.", "Index")
    answer = "Refund requests above 500 USD require manager approval. [chunk:vague]"
    result = grounding.verify(answer, [vague, APPROVALS, REFUNDS])

    assert "[chunk:approvals]" in result.clean_answer


def test_an_answer_that_is_mostly_ungrounded_still_fails_the_gate() -> None:
    answer = (
        "Enterprise plans include a dedicated success manager. [chunk:refunds] "
        "Onsite training is provided quarterly. [chunk:refunds] "
        "Customers may request a refund within 30 days. [chunk:refunds]"
    )
    result = grounding.verify(answer, [REFUNDS])

    assert not result.ok
    assert result.stripped_ratio > grounding.MAX_STRIPPED_RATIO


def test_the_refusal_sentinel_is_never_treated_as_an_ungrounded_answer() -> None:
    result = grounding.verify("INSUFFICIENT_CONTEXT", [REFUNDS])

    assert result.ok
    assert result.valid_chunk_ids == []
