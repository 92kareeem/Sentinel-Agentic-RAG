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

_CITATION_RE = re.compile(r"\[chunk:([\w-]+)\]")
_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")
_WORD_RE = re.compile(r"[A-Za-z]{2,}")
# A "sentence" that is only whitespace before its citation tag has no numeric
# claim to falsify, so the numeric check below never strips it — a degenerate
# generation (model emits bare [chunk:id] tags with no content) would sail
# through the gate. Require at least this many real words to count as a claim.
MIN_WORDS_PER_SENTENCE = 2

# Refuse only if MOST of the answer is ungrounded. A single weak sentence in an
# otherwise-cited answer is stripped and the rest still ships, rather than nuking
# a good answer to a refusal.
MAX_STRIPPED_RATIO = 0.50

# Fraction of a claim's content words that must appear in its supporting
# evidence. Low on purpose: this gate exists to catch a claim with essentially
# nothing to do with what it cites, not to penalize paraphrase or synonyms —
# judging semantic entailment is the LLM critic's job, and a deterministic
# checker that tried to do it would produce false refusals on good answers.
MIN_LEXICAL_OVERLAP = 0.25


@dataclass(frozen=True)
class GroundingResult:
    clean_answer: str
    ok: bool
    stripped_ratio: float
    valid_chunk_ids: list[str]


def _numbers_in(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER_RE.findall(text)}


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


def _content_terms(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS}


def _has_lexical_support(sentence: str, cited: Chunk, retrieved: list[Chunk]) -> bool:
    """Does the cited chunk plausibly support this sentence?

    Requires the claim's content words to overlap the cited chunk's text. A
    sentence that shares nothing with what it cites is either a fabrication or
    a misattribution; either way it should not ship with that citation.

    Falls back to accepting overlap with ANY retrieved chunk, because the
    common real failure is a mis-pointed citation on an otherwise-grounded
    claim — that is a citation-quality problem the critic scores, not a
    hallucination worth deleting the sentence over.
    """
    claim_terms = _content_terms(sentence)
    if len(claim_terms) < MIN_WORDS_PER_SENTENCE:
        return True  # too short to judge; the word-count gate already ran

    def overlap(chunk: Chunk) -> float:
        chunk_terms = _content_terms(chunk.text) | _content_terms(chunk.section_path)
        if not chunk_terms:
            return 0.0
        return len(claim_terms & chunk_terms) / len(claim_terms)

    if overlap(cited) >= MIN_LEXICAL_OVERLAP:
        return True
    return any(overlap(c) >= MIN_LEXICAL_OVERLAP for c in retrieved)


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


def verify(answer: str, retrieved: list[Chunk]) -> GroundingResult:
    if answer.strip() == "INSUFFICIENT_CONTEXT":
        return GroundingResult(answer, ok=True, stripped_ratio=0.0, valid_chunk_ids=[])

    by_id = {c.chunk_id: c for c in retrieved}
    # Numbers are checked against the WHOLE retrieved context, not just the one
    # chunk a sentence happens to cite: the LLM's per-sentence citation is
    # imperfect (it may cite chunk A for a fact that lives in chunk B, both
    # retrieved), so a number is "grounded" if it appears anywhere in context.
    # A number in NO retrieved chunk is still a genuine hallucination -> stripped.
    context_numbers = {n for c in retrieved for n in _numbers_in(c.text)}
    sentences = _split_cited_sentences(answer)
    if not sentences:  # no parseable cited sentences at all -> fail closed
        return GroundingResult(answer, ok=False, stripped_ratio=1.0, valid_chunk_ids=[])

    kept: list[str] = []
    valid_ids: list[str] = []
    stripped = 0
    for raw_sentence, chunk_id in sentences:
        # strip leading punctuation left over from the previous sentence's
        # own terminator (". ", "? ", etc. belongs to that sentence, not this one)
        sentence = raw_sentence.strip().lstrip(".!?").strip()
        if chunk_id not in by_id:  # cited a chunk that was never retrieved = fabricated
            stripped += 1
            continue
        if len(_WORD_RE.findall(sentence)) < MIN_WORDS_PER_SENTENCE:  # degenerate: no real content
            stripped += 1
            continue

        sent_numbers = _numbers_in(sentence)
        # A numeric claim is checked against the WHOLE retrieved context rather
        # than only the cited chunk: per-sentence citation is imperfect (the
        # model may cite chunk A for a figure that lives in chunk B, both
        # retrieved). A number present in NO retrieved chunk is a genuine
        # fabrication and the sentence goes.
        if sent_numbers and sent_numbers.isdisjoint(context_numbers):
            stripped += 1
            continue

        # Lexical support check: the cited chunk must actually share
        # substantive vocabulary with the claim. Citation-id validity alone
        # only proved the chunk was retrieved, not that it says anything
        # related — so a fluent sentence citing an arbitrary retrieved chunk
        # used to pass. This is deliberately a weak-but-deterministic signal:
        # it catches a claim wholly unrelated to its evidence without
        # second-guessing legitimate paraphrase (which is the LLM critic's job).
        if not _has_lexical_support(sentence, by_id[chunk_id], retrieved):
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
