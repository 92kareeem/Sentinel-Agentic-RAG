"""The gate's two jobs: is the claim supported, and by the passage it names?

Both failure modes produce answers that look correct. A wrong number reads as
authoritative; a wrong citation only breaks when someone clicks it. Every test
here runs offline: no model, no network.
"""

from __future__ import annotations

from sentinel_verify import MAX_STRIPPED_RATIO, ClaimStatus, Passage, RefusalReason, verify


def _p(pid: str, text: str, section: str = "Policy") -> Passage:
    return Passage(id=pid, text=text, section=section)


REFUNDS = _p(
    "refunds", "Customers may request a refund within 30 days of purchase.", "Refund policy"
)
HEADCOUNT = _p("headcount", "The support team grew to 30 employees across two regions.", "Team")
APPROVALS = _p(
    "approvals",
    "Refund requests above 500 USD require manager approval before processing.",
    "Refund approvals",
)
CHUNKS = [
    _p("c1", "Refunds take 3 days in the US with a 5% restocking fee."),
    _p("c2", "Engineers get a $2,400 laptop budget every 3 years."),
]


# ------------------------------------------------------------- basic outcomes


def test_good_answer_passes() -> None:
    v = verify("Refunds take 3 days [chunk:c1]. The budget is 2,400 dollars [chunk:c2].", CHUNKS)
    assert v.ok and v.passed and v.stripped_ratio == 0.0
    assert set(v.cited_ids) == {"c1", "c2"}
    assert all(c.status is ClaimStatus.SUPPORTED for c in v.claims)
    assert v.refusal is None


def test_fabricated_number_is_stripped_but_answer_ships() -> None:
    answer = (
        "Refunds take 3 days [chunk:c1]. Refunds take 99 days [chunk:c1]. Fee is 5% [chunk:c1]."
    )
    v = verify(answer, CHUNKS)
    assert "99" not in v.answer
    assert v.ok  # 1 of 3 stripped: the cleaned answer is served, not refused
    assert "3 days" in v.answer and "5%" in v.answer
    [bad] = v.stripped
    assert bad.unsupported_numbers == ("99",)


def test_mostly_fabricated_answer_refuses() -> None:
    answer = (
        "Refunds take 88 days [chunk:c1]. Fee is 77% [chunk:c1]. Refunds take 3 days [chunk:c1]."
    )
    v = verify(answer, CHUNKS)
    assert not v.ok and v.stripped_ratio > MAX_STRIPPED_RATIO
    assert v.refusal is RefusalReason.UNVERIFIABLE_ANSWER
    assert v.answer == ""  # a failed verdict never hands back the draft
    assert "guess" in v.safe_text()


def test_number_grounded_in_other_retrieved_passage_is_repaired() -> None:
    v = verify("The laptop budget is 2,400 dollars [chunk:c1].", CHUNKS)
    assert v.ok and "2,400" in v.answer
    assert "[chunk:c2]" in v.answer and v.repaired_citations == 1


def test_unknown_passage_id_is_stripped() -> None:
    v = verify("Something made up [chunk:ghost].", CHUNKS)
    assert not v.ok and v.stripped_ratio == 1.0


def test_model_refusal_passes_through_as_a_refusal() -> None:
    v = verify("INSUFFICIENT_CONTEXT", CHUNKS)
    assert v.ok and v.is_refusal and not v.passed
    assert v.refusal is RefusalReason.INSUFFICIENT_EVIDENCE
    assert v.cited_ids == ()


def test_uncited_answer_fails_closed() -> None:
    assert not verify("Trust me, refunds are instant.", CHUNKS).ok


def test_back_to_back_citations_with_no_prose_are_all_stripped() -> None:
    v = verify(" [chunk:c1] [chunk:c2]. [chunk:c1] [chunk:c2].", CHUNKS)
    assert not v.ok and v.stripped_ratio == 1.0 and v.answer == ""


# ------------------------------------------- numbers must live in the evidence


def test_a_number_from_an_unrelated_passage_no_longer_grounds_a_claim() -> None:
    """THE regression test. Numbers used to be checked against the union of
    every retrieved passage, so a fabricated refund window passed whenever any
    passage happened to contain the same digits: here, a headcount."""
    v = verify(
        "Customers may request a refund within 60 days of purchase. [chunk:refunds]",
        [REFUNDS, HEADCOUNT],
    )
    assert "60 days" not in v.answer


def test_the_same_claim_with_the_right_number_still_passes() -> None:
    v = verify(
        "Customers may request a refund within 30 days of purchase. [chunk:refunds]",
        [REFUNDS, HEADCOUNT],
    )
    assert v.ok and "30 days" in v.answer


def test_a_number_echoed_from_the_question_is_not_a_fabrication() -> None:
    v = verify(
        "At 20 days the request is inside the 30 day refund window. [chunk:refunds]",
        [REFUNDS],
        question="Can we refund at 20 days after purchase?",
    )
    assert v.ok and "20 days" in v.answer


def test_a_list_marker_is_not_read_as_a_factual_claim() -> None:
    v = verify("1. Customers may request a refund within 30 days. [chunk:refunds]", [REFUNDS])
    assert v.ok and "30 days" in v.answer


def test_a_figure_that_lives_only_in_the_section_heading_still_grounds() -> None:
    p = _p("s", "Requests must be filed in writing.", "Item 7.2 > Refunds")
    assert verify("Requests must be filed in writing under Item 7.2. [chunk:s]", [p]).ok


def test_thousands_separators_do_not_look_like_different_numbers() -> None:
    p = _p("c", "The annual cap is 1,200 USD per customer.", "Caps")
    assert verify("The annual cap is 1200 USD per customer. [chunk:c]", [p]).ok


# ----------------------------------------------- citations must point at truth


def test_a_citation_naming_the_wrong_passage_is_repaired_not_accepted() -> None:
    v = verify(
        "Refunds above 500 USD require manager approval. [chunk:refunds]", [REFUNDS, APPROVALS]
    )
    assert v.ok
    assert "[chunk:approvals]" in v.answer and "[chunk:refunds]" not in v.answer
    assert v.repaired_citations == 1 and v.cited_ids == ("approvals",)
    [claim] = v.claims
    assert claim.status is ClaimStatus.REPAIRED
    assert claim.cited == ("refunds",) and claim.supported_by == ("approvals",)


def test_a_correct_citation_is_left_alone_and_not_counted_as_repaired() -> None:
    v = verify(
        "Refunds above 500 USD require manager approval. [chunk:approvals]", [REFUNDS, APPROVALS]
    )
    assert v.repaired_citations == 0 and "[chunk:approvals]" in v.answer


def test_a_claim_nothing_retrieved_supports_is_still_stripped() -> None:
    v = verify(
        "Enterprise customers receive a dedicated success manager. [chunk:refunds]",
        [REFUNDS, HEADCOUNT],
    )
    assert "success manager" not in v.answer


def test_a_fabricated_id_can_be_repaired_when_the_claim_is_real() -> None:
    v = verify("Customers may request a refund within 30 days. [chunk:does-not-exist]", [REFUNDS])
    assert v.ok and "[chunk:refunds]" in v.answer


def test_repair_picks_the_passage_that_says_the_most_about_the_claim() -> None:
    vague = _p("vague", "Refunds are covered elsewhere in this handbook.", "Index")
    v = verify(
        "Refund requests above 500 USD require manager approval. [chunk:vague]",
        [vague, APPROVALS, REFUNDS],
    )
    assert "[chunk:approvals]" in v.answer


def test_an_answer_that_is_mostly_ungrounded_still_fails_the_gate() -> None:
    answer = (
        "Enterprise plans include a dedicated success manager. [chunk:refunds] "
        "Onsite training is provided quarterly. [chunk:refunds] "
        "Customers may request a refund within 30 days. [chunk:refunds]"
    )
    v = verify(answer, [REFUNDS])
    assert not v.ok and v.stripped_ratio > MAX_STRIPPED_RATIO


# ------------------------------------------ several passages cited at once

VENDOR = _p(
    "vendor",
    "The service helps banks compare document extraction vendors before procurement.",
    "Case study > Problem",
)
PILOT = _p(
    "pilot",
    "The recommended pilot runs for 90 days on a single document type.",
    "Case study > Plan",
)
TRUST = _p(
    "trust",
    "Buyers look for ISO 27001 certification and SOC 2 reports as trust signals.",
    "Case study > Trust",
)
SUMMARY = [VENDOR, PILOT, TRUST]
STACKED = " [chunk:vendor] [chunk:pilot] [chunk:trust]"


def test_a_summary_citing_several_passages_at_once_is_kept() -> None:
    answer = (
        "The case study describes a service that helps banks compare document extraction "
        "vendors before procurement. It recommends a pilot of 90 days on a single document "
        "type. Buyers look for ISO 27001 certification and SOC 2 reports." + STACKED
    )
    v = verify(answer, SUMMARY, question="what is this document about")
    assert v.ok and v.stripped_ratio == 0.0
    assert set(v.cited_ids) == {"vendor", "pilot", "trust"}


def test_each_sentence_keeps_only_the_passages_that_bear_on_it() -> None:
    answer = (
        "It recommends a pilot of 90 days on a single document type. "
        "Buyers look for ISO 27001 certification." + STACKED
    )
    v = verify(answer, SUMMARY)
    assert "90 days on a single document type [chunk:pilot]" in v.answer
    assert "ISO 27001 certification [chunk:trust]" in v.answer
    assert "[chunk:vendor]" not in v.answer


def test_an_invented_sentence_cannot_hide_inside_a_cited_paragraph() -> None:
    answer = (
        "The service helps banks compare document extraction vendors. "
        "The recommended pilot runs for 90 days. "
        "The company was founded in Lisbon in 2011 by a former central banker. "
        "Buyers look for ISO 27001 certification." + STACKED
    )
    v = verify(answer, SUMMARY)
    assert v.ok and v.stripped_ratio == 0.25
    assert "Lisbon" not in v.answer and "2011" not in v.answer


def test_a_figure_none_of_the_cited_passages_contain_is_still_stripped() -> None:
    v = verify(
        "The recommended pilot runs for 120 days on a single document type." + STACKED, SUMMARY
    )
    assert not v.ok and "120" not in v.answer


def test_a_fabricated_tag_in_a_group_is_ignored_not_trusted() -> None:
    v = verify("The recommended pilot runs for 90 days. [chunk:pilot] [chunk:made-up-id]", SUMMARY)
    assert v.ok and v.cited_ids == ("pilot",) and "made-up-id" not in v.answer


def test_a_short_fragment_is_not_scored_as_its_own_sentence() -> None:
    v = verify(
        "No. Customers may request a refund within 30 days of purchase [chunk:refunds].", [REFUNDS]
    )
    assert v.ok and v.stripped_ratio == 0.0


# ------------------------------------------ list-style answers

TABLE = _p(
    "casestudy_p0_s27_c0",
    "Risks include a long sales cycle, procurement delay and security rejection, "
    "each with a practical mitigation.",
    "Case study > Risks",
)


def test_a_passage_id_quoted_in_prose_is_not_read_as_a_numeric_claim() -> None:
    answer = (
        "The table in chunk casestudy_p0_s27_c0 lists risks such as a long sales cycle, "
        "procurement delay and security rejection. [chunk:casestudy_p0_s27_c0]"
    )
    v = verify(answer, [TABLE])
    assert v.ok and v.stripped_ratio == 0.0


def test_a_passage_id_never_reaches_the_reader() -> None:
    answer = (
        "Risks include procurement delay and security rejection "
        "(chunks casestudy_p0_s27_c0, _c1). [chunk:casestudy_p0_s27_c0]"
    )
    v = verify(answer, [TABLE])
    assert v.ok
    prose = v.answer.split("[chunk:")[0]
    assert "procurement delay and security rejection" in prose
    assert "casestudy" not in prose and "_c1" not in prose and "()" not in prose


def test_a_list_lead_in_is_not_counted_as_a_claim() -> None:
    answer = (
        "The main ones are:\n"
        "1. Customers may request a refund within 30 days of purchase [chunk:refunds].\n"
        "2. Refund requests above 500 USD require manager approval [chunk:approvals]."
    )
    v = verify(answer, [REFUNDS, APPROVALS])
    assert v.ok and v.stripped_ratio == 0.0


def test_an_answer_that_is_only_a_lead_in_fails_closed() -> None:
    v = verify("The main ones are: [chunk:refunds]", [REFUNDS])
    assert not v.ok and v.stripped_ratio == 1.0


# ------------------------------------------ citation shapes models emit


def test_lenticular_brackets_are_read_as_citations() -> None:
    v = verify(
        "Customers may request a refund within 30 days of purchase【chunk:refunds】", [REFUNDS]
    )
    assert v.ok and v.cited_ids == ("refunds",) and "【" not in v.answer


# ------------------------------------------ options


def test_the_threshold_is_adjustable() -> None:
    answer = "Refunds take 3 days [chunk:c1]. Refunds take 99 days [chunk:c1]."
    assert verify(answer, CHUNKS).ok  # exactly 50% stripped passes the default
    assert not verify(answer, CHUNKS, max_stripped_ratio=0.25).ok


def test_passages_without_ids_get_positional_ones() -> None:
    v = verify("Refunds take 3 days [chunk:p1].", [Passage(text=CHUNKS[0].text)])
    assert v.ok and v.cited_ids == ("p1",)
