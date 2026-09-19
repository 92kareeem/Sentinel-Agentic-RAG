"""Deterministic grounding verifier — the last gate before the user.

Role in architecture: the critic is an LLM and can be wrong; this check is
pure string logic and cannot hallucinate. It answers one question per
sentence: is there a retrieved passage that actually supports this, and is
the sentence pointing at it?

Three outcomes per sentence:
  * supported by the chunk it cites            -> kept as written
  * supported by a DIFFERENT retrieved chunk   -> citation repaired to that one
  * supported by nothing retrieved             -> stripped

If more than MAX_STRIPPED_RATIO of sentences are stripped the answer is
untrustworthy and is treated like a critic failure (repair or refuse).

What this deliberately does NOT do is judge semantic entailment. Lexical
overlap is a weak signal, and a deterministic checker that tried to be clever
about paraphrase would produce false refusals on good answers. Entailment is
the LLM critic's job; this gate exists to catch claims with no evidentiary
basis at all, and to make sure a citation points where it says it does.
"""

import re
from dataclasses import dataclass

from app.models.schemas import Chunk, is_insufficient_context

_CITATION_RE = re.compile(r"\[chunk:([\w-]+)\]")
_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")
_WORD_RE = re.compile(r"[A-Za-z]{2,}")
# Markdown the model adds for presentation. Stripped before numbers are
# extracted, so an enumerated answer ("1. The window is 30 days") does not
# have its own list marker read as a factual claim about the number one.
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")

# A "sentence" that is only whitespace before its citation tag has no claim to
# falsify, so the checks below would never strip it — a degenerate generation
# (bare [chunk:id] tags with no content) would sail through the gate. Require
# at least this many real words to count as a claim.
MIN_WORDS_PER_SENTENCE = 2

# Refuse only if MOST of the answer is ungrounded. A single weak sentence in an
# otherwise-cited answer is stripped and the rest still ships, rather than
# nuking a good answer to a refusal.
MAX_STRIPPED_RATIO = 0.50

# Fraction of a claim's content words that must appear in its supporting
# evidence. Low on purpose: see the module docstring.
MIN_LEXICAL_OVERLAP = 0.25


@dataclass(frozen=True)
class GroundingResult:
    clean_answer: str
    ok: bool
    stripped_ratio: float
    valid_chunk_ids: list[str]
    # Sentences whose citation pointed at a chunk that does not support them,
    # where another retrieved chunk does. Surfaced rather than silently
    # counted as a pass: a steady rate here means the synthesizer is
    # attributing badly, which is worth knowing even though the user now sees
    # a corrected citation.
    repaired_citations: int = 0


# Words carrying no topical signal — overlap on these says nothing about
# whether a chunk supports a claim.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "do", "does", "for", "from", "had", "has", "have", "if", "in", "into",
    "is", "it", "its", "may", "must", "not", "of", "on", "or", "should",
    "that", "the", "their", "there", "these", "this", "to", "was", "were",
    "will", "with", "you", "your", "we", "our", "they", "them", "he", "she",
    "his", "her",
})


def _numbers_in(text: str) -> set[str]:
    """Numeric tokens, comma separators removed so "1,200" matches "1200"."""
    return {n.replace(",", "").rstrip(".") for n in _NUMBER_RE.findall(text)}


def _content_terms(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS}


def _evidence_text(chunk: Chunk) -> str:
    """A chunk's text plus its section path.

    The path carries real evidence: "Item 7.2 > Refund window" grounds both
    the topic and the figure "7.2" that an answer may cite as a reference.
    """
    return f"{chunk.section_path}\n{chunk.text}"


def _overlap(claim_terms: set[str], chunk: Chunk) -> float:
    chunk_terms = _content_terms(_evidence_text(chunk))
    if not chunk_terms or not claim_terms:
        return 0.0
    return len(claim_terms & chunk_terms) / len(claim_terms)


def _supports(
    claim_terms: set[str],
    claim_numbers: set[str],
    chunk: Chunk,
    question_numbers: set[str],
) -> bool:
    """Does this specific chunk support this specific claim?

    Two independent conditions, and the numeric one is the point of the
    rewrite. Numbers used to be checked against the UNION of every retrieved
    chunk, so "the refund window is 30 days" passed whenever any retrieved
    passage happened to contain a 30 — a headcount, a page number, a different
    policy's threshold. For a product answering questions about refund
    windows, notice periods and limits, a plausible-but-wrong number is the
    most damaging thing it can say, and that check could not catch one.

    Numbers echoed from the user's own question are exempt. "Is 20 days within
    the window?" invites an answer that restates 20 while the evidence
    contains only 30; that restatement is not a fabrication, and stripping it
    would refuse a correct answer.
    """
    if _overlap(claim_terms, chunk) < MIN_LEXICAL_OVERLAP:
        return False
    unsupported = claim_numbers - _numbers_in(_evidence_text(chunk)) - question_numbers
    return not unsupported


def _split_cited_sentences(answer: str) -> list[tuple[str, str]]:
    """Pair each citation tag with the text that precedes it (since the prior
    tag, or the start of the answer).

    Walking tag-to-tag — rather than matching a sentence-boundary pattern like
    "[^.!?]+ [chunk:id]" — means a period anywhere in that preceding text (an
    abbreviation, a decimal, or simply two citations with no prose between
    them) can never break the parse. The old regex excluded '.' from the
    sentence body, so back-to-back citations like "[chunk:a] [chunk:b]."
    could make the whole span between them un-matchable — the citation was
    silently dropped from `sentences` entirely (neither kept nor counted as
    stripped), letting a degenerate, content-free answer read as fully
    grounded. Every character up to the last tag is accounted for here.
    """
    pairs: list[tuple[str, str]] = []
    pos = 0
    for m in _CITATION_RE.finditer(answer):
        pairs.append((answer[pos : m.start()], m.group(1)))
        pos = m.end()
    return pairs


def _best_supporting(
    claim_terms: set[str],
    claim_numbers: set[str],
    candidates: list[Chunk],
    question_numbers: set[str],
) -> Chunk | None:
    """The retrieved chunk that best supports this claim, or None.

    Reached only after the cited chunk has already failed. Picking the highest
    overlap rather than the first match matters: several passages can mention
    a topic, and the citation should land on the one that says the most about
    the claim, because a user will click it expecting to read exactly that.
    """
    supporting = [
        c for c in candidates if _supports(claim_terms, claim_numbers, c, question_numbers)
    ]
    if not supporting:
        return None
    return max(supporting, key=lambda c: _overlap(claim_terms, c))


def verify(answer: str, retrieved: list[Chunk], question: str = "") -> GroundingResult:
    if is_insufficient_context(answer):
        return GroundingResult(answer, ok=True, stripped_ratio=0.0, valid_chunk_ids=[])

    by_id = {c.chunk_id: c for c in retrieved}
    question_numbers = _numbers_in(question)
    sentences = _split_cited_sentences(answer)
    if not sentences:  # no parseable cited sentences at all -> fail closed
        return GroundingResult(answer, ok=False, stripped_ratio=1.0, valid_chunk_ids=[])

    kept: list[str] = []
    valid_ids: list[str] = []
    stripped = 0
    repaired = 0
    for raw_sentence, chunk_id in sentences:
        # Strip leading punctuation left over from the previous sentence's own
        # terminator (". ", "? " belong to that sentence, not this one).
        sentence = raw_sentence.strip().lstrip(".!?").strip()
        if len(_WORD_RE.findall(sentence)) < MIN_WORDS_PER_SENTENCE:
            stripped += 1  # degenerate: no real content to stand behind
            continue

        claim_terms = _content_terms(sentence)
        claim_numbers = _numbers_in(_LIST_MARKER_RE.sub("", sentence))

        cited = by_id.get(chunk_id)
        if cited is not None and _supports(claim_terms, claim_numbers, cited, question_numbers):
            attributed = cited
        else:
            # Either the cited id was never retrieved (a fabricated tag) or the
            # chunk it names does not support the claim. Before stripping, look
            # for a chunk that does.
            #
            # The old code accepted the sentence unchanged whenever ANY other
            # retrieved chunk supported it, leaving the wrong citation
            # attached. That reads as fine in the response body and breaks the
            # moment anyone clicks through: the passage shown does not contain
            # the claim, and a user who checks one citation and finds it wrong
            # stops trusting all the others. Re-pointing the tag costs nothing
            # and makes the citation true.
            found = _best_supporting(
                claim_terms, claim_numbers, retrieved, question_numbers
            )
            if found is None:
                stripped += 1
                continue
            attributed = found
            repaired += 1

        kept.append(f"{sentence.strip()} [chunk:{attributed.chunk_id}]")
        valid_ids.append(attributed.chunk_id)

    ratio = stripped / len(sentences)
    return GroundingResult(
        clean_answer=". ".join(kept) + ("." if kept else ""),
        ok=ratio <= MAX_STRIPPED_RATIO,
        stripped_ratio=ratio,
        valid_chunk_ids=valid_ids,
        repaired_citations=repaired,
    )
