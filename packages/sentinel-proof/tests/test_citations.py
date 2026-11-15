"""Citation styles: numeric [1] prompts and the tag shapes models drift into."""

from __future__ import annotations

from sentinel_proof import normalize_citations, passages_from_texts, strip_citations, verify
from sentinel_proof.citations import numeric_to_tags, tags_to_numeric

PASSAGES = passages_from_texts(
    [
        "Customers may request a refund within 30 days of purchase.",
        "Refund requests above 500 USD require manager approval before processing.",
        "The support team grew to 30 employees across two regions.",
    ]
)


def test_numeric_citations_are_checked_and_written_back_as_numbers() -> None:
    v = verify(
        "Customers may request a refund within 30 days [1]. "
        "Requests above 500 USD need manager approval [2].",
        PASSAGES,
        citation_style="numeric",
    )
    assert v.ok
    assert v.answer == (
        "Customers may request a refund within 30 days [1]. "
        "Requests above 500 USD need manager approval [2]."
    )
    assert v.cited_ids == ("p1", "p2")


def test_a_numeric_citation_pointing_at_the_wrong_passage_is_repaired() -> None:
    v = verify(
        "Requests above 500 USD need manager approval [1].", PASSAGES, citation_style="numeric"
    )
    assert v.ok and v.answer.endswith("[2].") and v.repaired_citations == 1


def test_a_numeric_group_is_one_group() -> None:
    v = verify(
        "Refunds are allowed within 30 days and need approval above 500 USD [1, 2].",
        PASSAGES,
        citation_style="numeric",
    )
    assert v.ok and v.answer.endswith("[1][2].")


def test_a_number_with_no_passage_behind_it_is_a_fabricated_citation() -> None:
    v = verify("The company was founded in 2011 [9].", PASSAGES, citation_style="numeric")
    assert not v.ok


def test_a_fabricated_figure_is_stripped_in_numeric_style_too() -> None:
    v = verify(
        "Customers may request a refund within 30 days [1]. "
        "Customers may request a refund within 60 days [1].",
        PASSAGES,
        citation_style="numeric",
    )
    assert v.ok and "60 days" not in v.answer and v.answer.endswith("[1].")


def test_bracketed_words_in_prose_are_not_citations() -> None:
    assert numeric_to_tags("as written [sic] in [1]", ["a"]) == "as written [sic] in [chunk:a]"


def test_roundtrip_through_numeric() -> None:
    ids = ["a", "b", "c"]
    assert tags_to_numeric(numeric_to_tags("x [1] y [2, 3]", ids), ids) == "x [1] y [2][3]"


def test_padded_and_fullwidth_tags_are_normalised() -> None:
    assert normalize_citations("a [ chunk: refunds ] b") == "a [chunk:refunds] b"
    assert normalize_citations("a ［chunk:refunds］ b") == "a [chunk:refunds] b"
    assert normalize_citations("a [chunk:refunds] b") == "a [chunk:refunds] b"


def test_normalising_does_not_invent_citations() -> None:
    text = "See section [3] and 【note】 for details."
    assert normalize_citations(text) == text


def test_strip_citations_leaves_clean_prose() -> None:
    assert strip_citations("Refunds take 30 days [chunk:a] [chunk:b]. Then more【chunk:c】.") == (
        "Refunds take 30 days. Then more."
    )
