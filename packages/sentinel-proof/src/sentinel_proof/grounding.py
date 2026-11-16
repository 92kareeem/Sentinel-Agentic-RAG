"""The deterministic grounding gate: the last check before a reader.

An LLM judge can be wrong. This is pure string logic and cannot hallucinate.
It answers one question per sentence: is there a retrieved passage that
actually supports this, and is the sentence pointing at it?

Three outcomes per sentence:

* supported by the passages it cites           -> kept as written
* supported by a DIFFERENT retrieved passage   -> citation repaired to that one
* supported by nothing retrieved               -> stripped

If more than ``max_stripped_ratio`` of the sentences are stripped the answer
is untrustworthy and the verdict is ``ok=False``: refuse, or repair upstream.

What this deliberately does NOT do is judge semantic entailment. Lexical
overlap is a weak signal, and a deterministic checker that tried to be clever
about paraphrase would produce false refusals on good answers. Entailment is
a judge's job; this gate catches claims with no evidentiary basis at all, and
makes sure every citation points where it says it does. In particular:
**every number in a claim must appear in a passage that claim cites.** For a
system answering questions about refund windows, notice periods and limits,
a plausible-but-wrong number is the most damaging thing it can say.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .citations import (
    CITATION_RE,
    CitationStyle,
    normalize_citations,
    numeric_to_tags,
    tags_to_numeric,
)
from .passages import Passage, _with_ids
from .refusal import RefusalReason, is_refusal, refusal_text

# Refuse only if MOST of the answer is ungrounded. A single weak sentence in an
# otherwise-cited answer is stripped and the rest still ships.
MAX_STRIPPED_RATIO = 0.50

# Fraction of a claim's content words that must appear in its evidence. Low on
# purpose: see the module docstring.
MIN_LEXICAL_OVERLAP = 0.25

# A "sentence" that is only whitespace before its citation has no claim to
# falsify, so a degenerate generation (bare tags, no content) would sail
# through. Require at least this many real words to count as a claim.
MIN_WORDS_PER_SENTENCE = 2

_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")
_WORD_RE = re.compile(r"[A-Za-z]{2,}")
# Markdown list markers. Stripped before numbers are extracted, so "1. The
# window is 30 days" is not read as a factual claim about the number one.
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
# What may sit between two tags for them to be one group: whitespace and the
# punctuation a model puts around stacked tags ("[a] [b]", "[a], [b].").
_BETWEEN_TAGS_RE = re.compile(r"^[\s.,;:]*$")
# A sentence ends at terminal punctuation followed by whitespace, or at a line
# break (list items and table rows are claims in their own right).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
# A passage id written INTO the prose rather than inside a tag: "the table in
# chunk a63a..._p0_s27_c0 lists...". Its digits would otherwise be read as
# figures the evidence must contain, and it means nothing to a reader anyway.
_CHUNK_REF_RE = re.compile(r"(?:\bchunks?\s*)?[\w-]*_(?:p\d+_)?s\d+_c\d+\b|(?<![\w])_c\d+\b")
_EMPTY_PARENS_RE = re.compile(r"\(\s*[,;\s]*\)")

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "must",
        "not",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "you",
        "your",
        "we",
        "our",
        "they",
        "them",
        "he",
        "she",
        "his",
        "her",
    }
)


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"  # kept, cited passages support it
    REPAIRED = "repaired"  # kept, citation re-pointed to the passage that supports it
    STRIPPED = "stripped"  # removed, nothing retrieved supports it

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Claim:
    """One sentence of the answer and what happened to it."""

    text: str
    status: ClaimStatus
    cited: tuple[str, ...]  # passage ids the model cited for it
    supported_by: tuple[str, ...]  # passage ids that actually support it (empty if stripped)
    unsupported_numbers: tuple[str, ...] = ()  # figures no cited passage contained


@dataclass(frozen=True)
class Verdict:
    """What the gate decided. ``ok`` is the one bit that matters; the rest explains it."""

    ok: bool
    answer: str  # the supported subset, re-cited; empty when nothing survived
    claims: tuple[Claim, ...]
    stripped_ratio: float
    cited_ids: tuple[str, ...]  # passages the surviving answer rests on, first-seen order
    repaired_citations: int
    refusal: RefusalReason | None  # set when ok is False, or when the model refused
    is_refusal: bool = False  # the model itself said the evidence was insufficient

    @property
    def stripped(self) -> tuple[Claim, ...]:
        return tuple(c for c in self.claims if c.status is ClaimStatus.STRIPPED)

    @property
    def passed(self) -> bool:
        """True when a verified answer exists. False for refusals of any kind."""
        return self.ok and not self.is_refusal

    def safe_text(self) -> str:
        """The verified answer, or refusal prose generated from the reason code.

        Never the rejected draft: that is the one text the gate just decided
        it could not stand behind.
        """
        if self.passed:
            return self.answer
        return refusal_text(self.refusal or RefusalReason.UNVERIFIABLE_ANSWER)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready view of the verdict: plain strings, lists and numbers.

        The same shape the CLI prints with ``--json`` and the playground
        renders, so a verdict logged from any of them reads the same.
        """
        d = asdict(self, dict_factory=_json_dict)
        d["refusal"] = self.refusal.value if self.refusal else None
        d["passed"] = self.passed
        d["safe_text"] = self.safe_text()
        return d


def _json_dict(items: list[tuple[str, Any]]) -> dict[str, Any]:
    """``asdict`` factory: tuples become lists and enums their values."""
    out: dict[str, Any] = {}
    for k, v in items:
        if isinstance(v, tuple):
            v = list(v)
        elif isinstance(v, Enum):
            v = v.value
        out[k] = v
    return out


@dataclass
class _Options:
    max_stripped_ratio: float = MAX_STRIPPED_RATIO
    min_lexical_overlap: float = MIN_LEXICAL_OVERLAP
    question_numbers: set[str] = field(default_factory=set)


# ------------------------------------------------------------------ helpers


def _numbers_in(text: str) -> set[str]:
    """Numeric tokens, comma separators removed so "1,200" matches "1200"."""
    return {n.replace(",", "").rstrip(".") for n in _NUMBER_RE.findall(text)}


def _content_terms(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS}


def _overlap(claim_terms: set[str], passage: Passage) -> float:
    terms = _content_terms(passage.evidence)
    if not terms or not claim_terms:
        return 0.0
    return len(claim_terms & terms) / len(claim_terms)


def _without_chunk_refs(sentence: str) -> str:
    text = _CHUNK_REF_RE.sub(" ", sentence)
    text = _EMPTY_PARENS_RE.sub("", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _is_lead_in(sentence: str) -> bool:
    """ "The main ones are:" introduces the claims that follow and asserts nothing."""
    return sentence.rstrip().endswith(":")


def _cited_units(answer: str) -> list[tuple[str, list[str]]]:
    """Pair each GROUP of citation tags with the text before it.

    Consecutive tags are one group: "...decisions. [a] [b] [c]" means these
    three passages together support what came before, which is how a summary
    drawing on several passages is naturally cited. Walking tag to tag means a
    period inside the text (an abbreviation, a decimal) can never make a tag
    unparseable, and every character before the last tag is accounted for.
    """
    units: list[tuple[str, list[str]]] = []
    pos = 0
    for m in CITATION_RE.finditer(answer):
        between = answer[pos : m.start()]
        if units and _BETWEEN_TAGS_RE.match(between):
            if m.group(1) not in units[-1][1]:
                units[-1][1].append(m.group(1))
        else:
            units.append((between, [m.group(1)]))
        pos = m.end()
    return units


def _sentences(text: str) -> list[str]:
    """The individual claims before a citation group, each checked on its own.

    Scoring a whole cited paragraph as one unit would let an invented sentence
    ride along inside an otherwise-true paragraph, diluted by its neighbours.
    A fragment too short to be a claim ("e.g.", "No.") is joined to the next
    piece rather than scored.
    """
    text = text.strip().lstrip(".!?").strip()
    out: list[str] = []
    carry = ""
    for piece in _SENTENCE_SPLIT_RE.split(text):
        piece = piece.strip()
        if not piece:
            continue
        if carry:
            piece = f"{carry} {piece}"
        if len(_WORD_RE.findall(piece)) < MIN_WORDS_PER_SENTENCE:
            carry = piece
            continue
        carry = ""
        out.append(piece)
    if carry and out:
        out[-1] = f"{out[-1]} {carry}"
    return out


def _supported_by_group(
    claim_terms: set[str], claim_numbers: set[str], passages: Sequence[Passage], opt: _Options
) -> bool:
    """Do the passages this claim cites, taken together, support it?

    Together, because that is what citing several passages asserts: the
    "90-day pilot" may come from one and "ISO 27001" from another. Every
    figure must still appear in something the claim actually cites.
    """
    if not passages or not claim_terms:
        return False
    evidence_terms: set[str] = set()
    evidence_numbers: set[str] = set()
    for p in passages:
        evidence_terms |= _content_terms(p.evidence)
        evidence_numbers |= _numbers_in(p.evidence)
    if len(claim_terms & evidence_terms) / len(claim_terms) < opt.min_lexical_overlap:
        return False
    return not (claim_numbers - evidence_numbers - opt.question_numbers)


def _minimal_support(
    claim_terms: set[str], claim_numbers: set[str], passages: Sequence[Passage], opt: _Options
) -> list[Passage] | None:
    """The smallest set of these passages that together support the claim.

    A group cited after a paragraph applies to every sentence in it, but each
    sentence should keep only the passages it rests on, so a reader who clicks
    a citation lands on text about THAT sentence. Greedy: take the passage
    that supplies the most missing figures, then the most new words, until
    the claim is supported. Deterministic, and exact for the two or three
    passages a sentence cites in practice.
    """
    if not _supported_by_group(claim_terms, claim_numbers, passages, opt):
        return None
    needed_numbers = claim_numbers - opt.question_numbers
    evidence = {p.id: (_content_terms(p.evidence), _numbers_in(p.evidence)) for p in passages}
    chosen: list[Passage] = []
    covered_terms: set[str] = set()
    covered_numbers: set[str] = set()
    remaining = list(passages)
    while not _supported_by_group(claim_terms, claim_numbers, chosen, opt):
        best = max(
            remaining,
            key=lambda p: (
                len((needed_numbers - covered_numbers) & evidence[p.id][1]),
                len((claim_terms - covered_terms) & evidence[p.id][0]),
            ),
        )
        chosen.append(best)
        remaining.remove(best)
        covered_terms |= evidence[best.id][0] & claim_terms
        covered_numbers |= evidence[best.id][1] & needed_numbers
    return chosen


def _best_supporting(
    claim_terms: set[str], claim_numbers: set[str], candidates: Sequence[Passage], opt: _Options
) -> Passage | None:
    """The retrieved passage that best supports this claim, or None.

    Reached only after the cited passages have failed. Highest overlap wins,
    because a reader will click the citation expecting to read exactly that.
    """
    supporting = [
        p for p in candidates if _supported_by_group(claim_terms, claim_numbers, [p], opt)
    ]
    if not supporting:
        return None
    return max(supporting, key=lambda p: _overlap(claim_terms, p))


# --------------------------------------------------------------------- gate


def verify(
    answer: str,
    passages: Sequence[Passage],
    question: str = "",
    *,
    citation_style: CitationStyle = "tag",
    max_stripped_ratio: float = MAX_STRIPPED_RATIO,
    min_lexical_overlap: float = MIN_LEXICAL_OVERLAP,
) -> Verdict:
    """Check an answer against the passages it was generated from.

    ``question`` matters: numbers echoed from it are not fabrications. "Is 20
    days within the window?" invites an answer that restates 20 while the
    evidence contains only 30, and stripping that would refuse a correct answer.

    ``citation_style="numeric"`` reads ``[1]``, ``[2, 3]`` as 1-based positions
    in ``passages`` and writes the verified answer back in that style.
    """
    passages = _with_ids(passages)
    ids = [p.id for p in passages]
    opt = _Options(max_stripped_ratio, min_lexical_overlap, _numbers_in(question))

    text = normalize_citations(answer)
    if citation_style == "numeric":
        text = numeric_to_tags(text, ids)

    if is_refusal(text):
        return Verdict(
            ok=True,
            answer="",
            claims=(),
            stripped_ratio=0.0,
            cited_ids=(),
            repaired_citations=0,
            refusal=RefusalReason.INSUFFICIENT_EVIDENCE,
            is_refusal=True,
        )

    by_id = {p.id: p for p in passages}
    units = _cited_units(text)
    if not units:  # no parseable citations at all: fail closed
        return _failed(text, (), 1.0)

    claims: list[Claim] = []
    kept: list[str] = []
    cited_ids: list[str] = []
    stripped = repaired = total = 0

    for prose, tag_ids in units:
        # Tags naming passages that were never retrieved are fabricated; they
        # cannot support anything, so they simply drop out of the group.
        cited = [by_id[i] for i in tag_ids if i in by_id]
        sentences = _sentences(prose)
        if not sentences:
            total += 1
            stripped += 1  # degenerate: tags with nothing to stand behind
            claims.append(Claim(prose.strip(" .,;:\n\t"), ClaimStatus.STRIPPED, tuple(tag_ids), ()))
            continue

        for raw in sentences:
            sentence = _without_chunk_refs(raw)
            if _is_lead_in(sentence) or len(_WORD_RE.findall(sentence)) < MIN_WORDS_PER_SENTENCE:
                continue  # not a claim; counts neither for nor against
            total += 1
            terms = _content_terms(sentence)
            numbers = _numbers_in(_LIST_MARKER_RE.sub("", sentence))

            support = _minimal_support(terms, numbers, cited, opt)
            status = ClaimStatus.SUPPORTED
            if support is None:
                # The passages it cites do not support it (or were never
                # retrieved). Before stripping, look for one that does, and
                # re-point the citation so that clicking it lands on the
                # claim. Accepting the sentence with its wrong tag would read
                # fine and break the moment anyone checked.
                found = _best_supporting(terms, numbers, passages, opt)
                if found is None:
                    stripped += 1
                    missing = tuple(sorted(numbers - opt.question_numbers - _all_numbers(cited)))
                    claims.append(
                        Claim(sentence, ClaimStatus.STRIPPED, tuple(tag_ids), (), missing)
                    )
                    continue
                support = [found]
                status = ClaimStatus.REPAIRED
                repaired += 1

            # In retrieval order, not the order the greedy cover found them,
            # so "[1][2]" never comes out as "[2][1]" for the same evidence.
            support.sort(key=lambda p: ids.index(p.id))
            tags = " ".join(f"[chunk:{p.id}]" for p in support)
            kept.append(f"{sentence.rstrip('.').strip()} {tags}")
            claims.append(Claim(sentence, status, tuple(tag_ids), tuple(p.id for p in support)))
            for p in support:
                if p.id not in cited_ids:
                    cited_ids.append(p.id)

    if total == 0:
        # Nothing but lead-ins and passage references: no claim was made, so
        # there is nothing to stand behind. Fail closed, as for no citations.
        return _failed(text, tuple(claims), 1.0)

    ratio = stripped / total
    clean = ". ".join(kept) + ("." if kept else "")
    if citation_style == "numeric":
        clean = tags_to_numeric(clean, ids)
    ok = ratio <= opt.max_stripped_ratio
    return Verdict(
        ok=ok,
        # A failed verdict carries no text at all. The surviving sentences are
        # still in ``claims`` for diagnosis, but handing back a partial answer
        # the gate just judged untrustworthy invites a caller to ship it.
        answer=clean if ok else "",
        claims=tuple(claims),
        stripped_ratio=ratio,
        cited_ids=tuple(cited_ids),
        repaired_citations=repaired,
        refusal=None if ok else RefusalReason.UNVERIFIABLE_ANSWER,
    )


def _all_numbers(passages: Sequence[Passage]) -> set[str]:
    out: set[str] = set()
    for p in passages:
        out |= _numbers_in(p.evidence)
    return out


def _failed(text: str, claims: tuple[Claim, ...], ratio: float) -> Verdict:
    return Verdict(
        ok=False,
        answer="",
        claims=claims,
        stripped_ratio=ratio,
        cited_ids=(),
        repaired_citations=0,
        refusal=RefusalReason.UNVERIFIABLE_ANSWER,
    )
