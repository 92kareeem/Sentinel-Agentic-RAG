"""Refusing is an outcome, with a code, never a guess.

Two ideas from running this in production:

* The model is told to emit one exact token, ``INSUFFICIENT_CONTEXT``, when
  the evidence does not answer the question. :func:`is_refusal` recognises it
  however the model decorates it (quotes, a period, a stray citation tag,
  wrong case), as long as it is the WHOLE answer.
* The text shown for a refusal is generated from the reason code, never taken
  from the model. A refusal must not carry the draft it rejected: that draft
  is the one text the system just decided it could not stand behind.
"""

from __future__ import annotations

import re
from enum import Enum

REFUSAL_SENTINEL = "INSUFFICIENT_CONTEXT"

_TAG_RE = re.compile(r"\[(?:chunk:)?[\w-]+\]")
_SENTINEL_RE = re.compile(rf"^[\s\W]*{REFUSAL_SENTINEL}[\s\W]*$", re.IGNORECASE)


def is_refusal(answer: str) -> bool:
    """Is this answer the refusal sentinel rather than a real answer?"""
    return bool(_SENTINEL_RE.match(_TAG_RE.sub("", answer)))


class RefusalReason(str, Enum):
    """Why no answer was delivered, as something a UI or a log can act on."""

    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # the passages do not answer it
    UNVERIFIABLE_ANSWER = "UNVERIFIABLE_ANSWER"  # drafted, but the evidence does not support it
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"  # a time, token or attempt cap fired first

    def __str__(self) -> str:
        return self.value


_TEXT: dict[RefusalReason, str] = {
    RefusalReason.INSUFFICIENT_EVIDENCE: (
        "I couldn't find anything in the provided documents that answers this question."
    ),
    RefusalReason.UNVERIFIABLE_ANSWER: (
        "I found related material, but couldn't confirm an answer was fully "
        "supported by it, so I'm not going to guess."
    ),
    RefusalReason.BUDGET_EXHAUSTED: (
        "This question hit the processing limit before a verified answer was "
        "ready. Please try again."
    ),
}


def refusal_text(reason: RefusalReason) -> str:
    """Prose that is safe to show for a reason code. No model output reaches it."""
    return _TEXT[reason]
