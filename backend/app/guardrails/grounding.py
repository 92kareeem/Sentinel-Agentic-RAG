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

# The same tag in the other shapes the model emits. gpt-oss is trained on
# OpenAI's own citation style and, especially the 120b model, often writes
# 【chunk:id】 (CJK lenticular brackets) instead of [chunk:id], or pads it:
# "[ chunk: id ]". None of those matched the parser, so a correct, fully
# cited answer read as UNCITED, failed closed, and was refused — measured at
# 4 refusals in 5 runs on a golden question the router sends to 120b. The
# critic rated every one of those drafts accurate.
_ALT_CITATION_RE = re.compile(r"[\[【［〔]\s*chunk\s*:\s*([\w-]+)\s*[\]】］〕]")


def normalize_citations(text: str) -> str:
    """Rewrite every recognised citation shape to the canonical [chunk:id].

    Called on the synthesizer's raw output, so the UI, the history and this
    gate all see one format; and again in verify(), because the gate is the
    last line before the user and must not depend on what ran before it.
    """
    return _ALT_CITATION_RE.sub(r"[chunk:\1]", text)
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


# A chunk id written INTO the prose rather than inside a [chunk:...] tag:
# "the table in chunk a63a...142_p0_s27_c0 lists...", or shorthand like
# "(chunks ..._s25_c1, _c2, _c3)". The model does this on list-style answers.
# Left in, the id's digits (63, 074, 7390...) were read as factual NUMBERS the
# evidence had to contain, and four well-supported sentences — 63-82% word
# overlap — were stripped for "unsupported figures", refusing a correct
# answer. They are also meaningless hashes to a reader, so they come out of
# the text that is shipped, not just out of the analysis.
_CHUNK_REF_RE = re.compile(
    r"(?:\bchunks?\s*)?[\w-]*_(?:p\d+_)?s\d+_c\d+\b"  # full id, optionally "chunk <id>"
    r"|(?<![\w])_c\d+\b"  # shorthand continuation: "_c2, _c3"
)
_EMPTY_PARENS_RE = re.compile(r"\(\s*[,;\s]*\)")


def _without_chunk_refs(sentence: str) -> str:
    text = _CHUNK_REF_RE.sub(" ", sentence)
    text = _EMPTY_PARENS_RE.sub("", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)  # "table , lists" -> "table, lists"
    return re.sub(r"\s{2,}", " ", text).strip()


def _is_lead_in(sentence: str) -> bool:
    """ "The main ones are:" — introduces the claims that follow and asserts
    nothing itself. Scoring it would strip it as unsupported and count that
    against the answer, so list-style answers were penalised for having the
    sentence that makes a list readable."""
    return sentence.rstrip().endswith(":")


# What may sit between two tags for them to be one group: whitespace and the
# punctuation a model puts around stacked tags ("[a] [b]", "[a], [b].").
_BETWEEN_TAGS_RE = re.compile(r"^[\s.,;:]*$")
# A sentence ends at terminal punctuation followed by whitespace, or at a line
# break (list items and table rows are claims in their own right).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _cited_units(answer: str) -> list[tuple[str, list[str]]]:
    """Pair each GROUP of citation tags with the text that precedes it.

    Consecutive tags are one group. "...procurement decisions. [chunk:a]
    [chunk:b] [chunk:c]" means "these three passages together support what
    came before" — the normal way to cite a summary, which draws on several
    passages at once. This used to be parsed tag by tag, so the paragraph was
    paired with [a] alone and [b] and [c] each with an EMPTY sentence. The
    empties were counted as stripped, the paragraph was scored against one
    passage when its facts were spread over three, and a correct summary —
    rated faithful by the LLM critic — was refused. "What is this document
    about?", one of the example questions the UI itself suggests, failed
    every time.

    Walking tag to tag, rather than matching a sentence pattern, is kept from
    the earlier fix: a period anywhere in the preceding text (an abbreviation,
    a decimal) can never make a tag unparseable, and every character before
    the last tag is accounted for. Bare tags with no prose still form a group
    with no text, which verify() counts as a stripped, content-free claim.
    """
    units: list[tuple[str, list[str]]] = []
    pos = 0
    for m in _CITATION_RE.finditer(answer):
        between = answer[pos : m.start()]
        if units and _BETWEEN_TAGS_RE.match(between):
            if m.group(1) not in units[-1][1]:
                units[-1][1].append(m.group(1))
        else:
            units.append((between, [m.group(1)]))
        pos = m.end()
    return units


def _sentences(text: str) -> list[str]:
    """The individual claims in the text before a citation group.

    Each is checked on its own. Scoring a whole cited paragraph as one unit
    would let an invented sentence ride along inside an otherwise-true
    paragraph, diluted by its neighbours' overlap.

    A fragment too short to be a claim — "e.g.", "No." split off by the
    sentence pattern — is joined to the next piece rather than scored: an
    abbreviation must not count as a stripped sentence.
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
    claim_terms: set[str],
    claim_numbers: set[str],
    chunks: list[Chunk],
    question_numbers: set[str],
) -> bool:
    """Do the passages this claim cites, taken together, support it?

    Together, because that is what citing several passages asserts: the
    "90-day MVP" may come from one and "ISO 27001" from another. The numeric
    rule is unchanged — every figure must appear in something the claim
    actually cites — it is just not confined to the first tag of the group.
    With a single citation this is exactly _supports().
    """
    if not chunks or not claim_terms:
        return False
    evidence_terms: set[str] = set()
    evidence_numbers: set[str] = set()
    for c in chunks:
        evidence = _evidence_text(c)
        evidence_terms |= _content_terms(evidence)
        evidence_numbers |= _numbers_in(evidence)
    if len(claim_terms & evidence_terms) / len(claim_terms) < MIN_LEXICAL_OVERLAP:
        return False
    return not (claim_numbers - evidence_numbers - question_numbers)


def _minimal_support(
    claim_terms: set[str],
    claim_numbers: set[str],
    chunks: list[Chunk],
    question_numbers: set[str],
) -> list[Chunk] | None:
    """The smallest set of these passages that together support the claim,
    or None if even all of them do not.

    A group cited after a paragraph applies to every sentence in it, but each
    sentence should keep only the passages it actually rests on, so a reader
    who clicks a citation lands on text about THAT sentence. "Shares any word
    with it" is not that test — "document" appears in half of any corpus and
    would tag a sentence about a 90-day pilot with a passage about vendors.

    Greedy: take the passage that supplies the most missing figures, then the
    most new words, until the claim is supported. Greedy set cover is not
    always minimal in theory; for the two or three passages a sentence cites,
    it is exact in practice, and it is deterministic.
    """
    if not _supported_by_group(claim_terms, claim_numbers, chunks, question_numbers):
        return None
    needed_numbers = claim_numbers - question_numbers
    evidence = {
        c.chunk_id: (_content_terms(_evidence_text(c)), _numbers_in(_evidence_text(c)))
        for c in chunks
    }
    chosen: list[Chunk] = []
    covered_terms: set[str] = set()
    covered_numbers: set[str] = set()
    remaining = list(chunks)
    while not _supported_by_group(claim_terms, claim_numbers, chosen, question_numbers):
        best = max(
            remaining,
            key=lambda c: (
                len((needed_numbers - covered_numbers) & evidence[c.chunk_id][1]),
                len((claim_terms - covered_terms) & evidence[c.chunk_id][0]),
            ),
        )
        chosen.append(best)
        remaining.remove(best)
        covered_terms |= evidence[best.chunk_id][0] & claim_terms
        covered_numbers |= evidence[best.chunk_id][1] & needed_numbers
    return chosen


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
    answer = normalize_citations(answer)
    if is_insufficient_context(answer):
        return GroundingResult(answer, ok=True, stripped_ratio=0.0, valid_chunk_ids=[])

    by_id = {c.chunk_id: c for c in retrieved}
    question_numbers = _numbers_in(question)
    units = _cited_units(answer)
    if not units:  # no parseable citations at all -> fail closed
        return GroundingResult(answer, ok=False, stripped_ratio=1.0, valid_chunk_ids=[])

    kept: list[str] = []
    valid_ids: list[str] = []
    stripped = 0
    repaired = 0
    total = 0
    for text, cited_ids in units:
        # Tags naming chunks that were never retrieved are fabricated; they
        # cannot support anything, so they simply drop out of the group.
        cited = [by_id[i] for i in cited_ids if i in by_id]
        claims = _sentences(text)
        if not claims:
            total += 1
            stripped += 1  # degenerate: tags with no real content to stand behind
            continue

        for raw_sentence in claims:
            sentence = _without_chunk_refs(raw_sentence)
            if _is_lead_in(sentence) or len(_WORD_RE.findall(sentence)) < MIN_WORDS_PER_SENTENCE:
                # A lead-in, or a "sentence" that was nothing but a chunk
                # reference. Neither is a claim; neither counts for or against.
                continue
            total += 1
            claim_terms = _content_terms(sentence)
            claim_numbers = _numbers_in(_LIST_MARKER_RE.sub("", sentence))

            support = _minimal_support(claim_terms, claim_numbers, cited, question_numbers)
            if support is not None:
                sources = support
            else:
                # The passages it cites do not support it (or were never
                # retrieved). Before stripping, look for one that does.
                #
                # Accepting the sentence with its original tag whenever ANY
                # retrieved chunk supported it — as this once did — reads fine
                # in the response and breaks the moment anyone clicks through:
                # the passage shown does not contain the claim, and a reader
                # who finds one wrong citation stops trusting all the others.
                # Re-pointing the tag costs nothing and makes it true.
                found = _best_supporting(claim_terms, claim_numbers, retrieved, question_numbers)
                if found is None:
                    stripped += 1
                    continue
                sources = [found]
                repaired += 1

            tags = " ".join(f"[chunk:{c.chunk_id}]" for c in sources)
            kept.append(f"{sentence.rstrip('.').strip()} {tags}")
            for c in sources:
                if c.chunk_id not in valid_ids:
                    valid_ids.append(c.chunk_id)

    if total == 0:
        # Nothing but lead-ins and chunk references: no claim was made, so
        # there is nothing to stand behind. Fail closed, as for no citations.
        return GroundingResult(answer, ok=False, stripped_ratio=1.0, valid_chunk_ids=[])
    ratio = stripped / total
    return GroundingResult(
        clean_answer=". ".join(kept) + ("." if kept else ""),
        ok=ratio <= MAX_STRIPPED_RATIO,
        stripped_ratio=ratio,
        valid_chunk_ids=valid_ids,
        repaired_citations=repaired,
    )
