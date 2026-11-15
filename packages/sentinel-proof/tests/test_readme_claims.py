"""The README's examples and FAQ answers, as tests.

The README is what people copy and what gets quoted back to them, so every
behaviour it promises is pinned here. If one of these fails, fix the code or
fix the README, never just the test.
"""

from sentinel_proof import REFUSAL_SENTINEL, ClaimStatus, passages_from_texts, verify

REFUND = passages_from_texts(
    [
        "Customers may request a refund within 30 days of purchase.",
        "Refund requests above 500 USD require manager approval.",
    ]
)

PRICING = passages_from_texts(
    [
        "The premium plan costs $1,200 per year with a 4.5% discount for annual billing.",
        "Support is available on weekdays.",
    ]
)


def test_quickstart_example() -> None:
    answer = (
        "Customers can get a refund within 30 days [1]. "
        "Requests above 500 USD need manager approval [1]. "
        "Refunds are processed in 2 business days [2]."
    )
    v = verify(answer, REFUND, citation_style="numeric")

    assert v.ok
    assert v.answer == (
        "Customers can get a refund within 30 days [1]. "
        "Requests above 500 USD need manager approval [2]."
    )
    assert v.repaired_citations == 1
    assert [c.text for c in v.stripped] == ["Refunds are processed in 2 business days"]
    assert v.safe_text() == v.answer


def test_currency_and_decimal_figures_are_checked() -> None:
    good = "The premium plan costs $1,200 per year [1]. Annual billing gets a 4.5% discount [1]."
    assert verify(good, PRICING, citation_style="numeric").stripped == ()

    bad = (
        "The premium plan costs $1,300 per year [1]. "
        "Annual billing gets a 4.5% discount [1]. Support works weekdays [2]."
    )
    v = verify(bad, PRICING, citation_style="numeric")
    assert v.ok
    assert [c.text for c in v.stripped] == ["The premium plan costs $1,300 per year"]
    assert "1,300" not in v.answer


def test_citation_to_unretrieved_passage_is_repointed() -> None:
    v = verify(
        "Annual billing for the premium plan gets a 4.5% discount [9].",
        PRICING,
        citation_style="numeric",
    )
    assert v.ok
    assert v.repaired_citations == 1
    assert v.answer.endswith("[1].")
    assert v.claims[0].status is ClaimStatus.REPAIRED


def test_mostly_unsupported_answer_is_refused_without_leaking_the_draft() -> None:
    v = verify(
        "The premium plan costs $9,999 per year [1]. It has a 7% discount [1].",
        PRICING,
        citation_style="numeric",
    )
    assert not v.ok
    assert v.answer == ""
    assert v.refusal is not None and v.refusal.value == "UNVERIFIABLE_ANSWER"
    assert "9,999" not in v.safe_text()
    assert "7%" not in v.safe_text()


def test_numbers_from_the_question_are_allowed() -> None:
    v = verify(
        "The premium plan costs 1,300 per year [1].",
        PRICING,
        question="Is 1,300 the price of the premium plan?",
        citation_style="numeric",
    )
    assert v.ok


def test_model_refusal_passes_through_with_a_reason() -> None:
    v = verify(REFUSAL_SENTINEL, PRICING, citation_style="numeric")
    assert v.is_refusal
    assert v.refusal is not None and v.refusal.value == "INSUFFICIENT_EVIDENCE"
