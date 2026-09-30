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


# ------------------------------------------ several passages cited at once
#
# Found in manual testing: "What is this document about?" — one of the example
# questions the UI itself suggests — was refused every time. The model wrote a
# correct summary and cited it the natural way, all tags stacked at the end:
#
#     "...one paragraph of four sentences... [chunk:a] [chunk:b] [chunk:c]"
#
# The gate paired the paragraph with [a] alone and [b], [c] with EMPTY
# sentences, stripped all of them, and refused an answer the LLM critic had
# rated fully faithful.

VENDOR = _chunk(
    "vendor",
    "The service helps banks compare document extraction vendors before procurement.",
    section="Case study > Problem",
)
PILOT = _chunk(
    "pilot",
    "The recommended pilot runs for 90 days on a single document type.",
    section="Case study > Plan",
)
TRUST = _chunk(
    "trust",
    "Buyers look for ISO 27001 certification and SOC 2 reports as trust signals.",
    section="Case study > Trust",
)
SUMMARY_CHUNKS = [VENDOR, PILOT, TRUST]
STACKED = " [chunk:vendor] [chunk:pilot] [chunk:trust]"


def test_a_summary_citing_several_passages_at_once_is_kept() -> None:
    answer = (
        "The case study describes a service that helps banks compare document extraction "
        "vendors before procurement. It recommends a pilot of 90 days on a single document "
        "type. Buyers look for ISO 27001 certification and SOC 2 reports." + STACKED
    )

    r = grounding.verify(answer, SUMMARY_CHUNKS, question="what is this document about")

    assert r.ok
    assert r.stripped_ratio == 0.0
    assert set(r.valid_chunk_ids) == {"vendor", "pilot", "trust"}


def test_each_sentence_keeps_only_the_passages_that_bear_on_it() -> None:
    """A reader clicking the citation after "90 days" should land on the plan,
    not on a list of every passage the paragraph drew from."""
    answer = (
        "It recommends a pilot of 90 days on a single document type. "
        "Buyers look for ISO 27001 certification." + STACKED
    )

    r = grounding.verify(answer, SUMMARY_CHUNKS)

    assert "90 days on a single document type [chunk:pilot]" in r.clean_answer
    assert "ISO 27001 certification [chunk:trust]" in r.clean_answer
    assert "[chunk:vendor]" not in r.clean_answer


def test_an_invented_sentence_cannot_hide_inside_a_cited_paragraph() -> None:
    """The reason each sentence is still checked separately: scoring the whole
    paragraph as one claim would let this fabrication ride along, diluted by
    three true neighbours."""
    answer = (
        "The service helps banks compare document extraction vendors. "
        "The recommended pilot runs for 90 days. "
        "The company was founded in Lisbon in 2011 by a former central banker. "
        "Buyers look for ISO 27001 certification." + STACKED
    )

    r = grounding.verify(answer, SUMMARY_CHUNKS)

    assert r.ok  # one of four stripped — below the refusal threshold
    assert r.stripped_ratio == 0.25
    assert "Lisbon" not in r.clean_answer
    assert "2011" not in r.clean_answer


def test_a_figure_none_of_the_cited_passages_contain_is_still_stripped() -> None:
    """Group support widens WHERE a figure may come from to every passage the
    sentence cites — never to passages it does not."""
    answer = "The recommended pilot runs for 120 days on a single document type." + STACKED

    r = grounding.verify(answer, SUMMARY_CHUNKS)

    assert not r.ok
    assert "120" not in r.clean_answer


def test_a_fabricated_tag_in_a_group_is_ignored_not_trusted() -> None:
    answer = "The recommended pilot runs for 90 days. [chunk:pilot] [chunk:made-up-id]"

    r = grounding.verify(answer, SUMMARY_CHUNKS)

    assert r.ok
    assert r.valid_chunk_ids == ["pilot"]
    assert "made-up-id" not in r.clean_answer


def test_a_short_fragment_is_not_scored_as_its_own_sentence() -> None:
    """ "No." split off by the sentence pattern is part of the claim that
    follows, not a separate one-word claim to strip."""
    answer = "No. Customers may request a refund within 30 days of purchase [chunk:refunds]."

    r = grounding.verify(answer, [REFUNDS])

    assert r.ok
    assert r.stripped_ratio == 0.0


# ------------------------------------------ list-style answers
#
# Also found in manual testing, on "What are the main policies or requirements
# described here?" (another of the UI's suggested questions): the critic rated
# the answer 0.9, grounding stripped 57% of it. Two causes, both below.

TABLE = _chunk(
    "casestudy_p0_s27_c0",
    "Risks include a long sales cycle, procurement delay and security rejection, "
    "each with a practical mitigation.",
    section="Case study > Risks",
)


def test_a_chunk_id_quoted_in_prose_is_not_read_as_a_numeric_claim() -> None:
    """ "The table in chunk casestudy_p0_s27_c0 lists..." — the model wrote the
    id into the sentence. Its digits (0, 27) were read as figures the evidence
    had to contain, and a sentence with 80% word overlap was stripped."""
    answer = (
        "The table in chunk casestudy_p0_s27_c0 lists risks such as a long sales cycle, "
        "procurement delay and security rejection. [chunk:casestudy_p0_s27_c0]"
    )

    r = grounding.verify(answer, [TABLE])

    assert r.ok
    assert r.stripped_ratio == 0.0


def test_a_chunk_id_never_reaches_the_reader() -> None:
    """A 40-character hash in the middle of an answer means nothing to the
    person reading it; the citation chip is how they reach the source."""
    answer = (
        "Risks include procurement delay and security rejection "
        "(chunks casestudy_p0_s27_c0, _c1). [chunk:casestudy_p0_s27_c0]"
    )

    r = grounding.verify(answer, [TABLE])

    # The sentence must SURVIVE, cleaned. Asserting only that the id is absent
    # would pass just as well if the whole sentence were stripped — which is
    # exactly what happened before this fix.
    assert r.ok
    prose = r.clean_answer.split("[chunk:")[0]
    assert "procurement delay and security rejection" in prose
    assert "casestudy" not in prose
    assert "_c1" not in prose
    assert "()" not in prose


def test_a_list_lead_in_is_not_counted_as_a_claim() -> None:
    """ "The main ones are:" asserts nothing; it introduces what follows.
    Counting it as an unsupported claim penalised every list-style answer."""
    answer = (
        "The main ones are:\n"
        "1. Customers may request a refund within 30 days of purchase [chunk:refunds].\n"
        "2. Refund requests above 500 USD require manager approval [chunk:approvals]."
    )

    r = grounding.verify(answer, [REFUNDS, APPROVALS])

    assert r.ok
    assert r.stripped_ratio == 0.0


def test_an_answer_that_is_only_a_lead_in_fails_closed() -> None:
    """No claim at all is not a grounded answer — and must not divide by zero."""
    r = grounding.verify("The main ones are: [chunk:refunds]", [REFUNDS])

    assert not r.ok
    assert r.stripped_ratio == 1.0


# ------------------------------------------ citation shapes the model emits
#
# gpt-oss (especially 120b) often writes OpenAI's native 【chunk:id】 instead of
# [chunk:id]. Unrecognised, a correct and fully cited answer read as uncited
# and was refused: 4 of 5 runs of a golden question the router sends to 120b.


def test_lenticular_brackets_are_read_as_citations() -> None:
    answer = "Customers may request a refund within 30 days of purchase【chunk:refunds】"

    r = grounding.verify(answer, [REFUNDS])

    assert r.ok
    assert r.valid_chunk_ids == ["refunds"]
    assert "【" not in r.clean_answer


def test_padded_and_fullwidth_citations_are_normalised() -> None:
    assert grounding.normalize_citations("a [ chunk: refunds ] b") == "a [chunk:refunds] b"
    assert grounding.normalize_citations("a ［chunk:refunds］ b") == "a [chunk:refunds] b"
    assert grounding.normalize_citations("a [chunk:refunds] b") == "a [chunk:refunds] b"


def test_normalising_does_not_invent_citations() -> None:
    """Only the chunk: form is rewritten; ordinary bracketed text is left alone."""
    text = "See section [3] and 【note】 for details."
    assert grounding.normalize_citations(text) == text
