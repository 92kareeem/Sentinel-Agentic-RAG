"""Citation tags: the shapes models write them in, and one canonical form.

Internally every citation is ``[chunk:<passage id>]``. Two things feed that:

* **Tag style** (``[chunk:id]``), which models also write as ``【chunk:id】``
  (gpt-oss, trained on OpenAI's own citation glyphs), ``[ chunk: id ]`` or
  with full-width brackets. All are normalised; none of them should turn a
  correct, fully cited answer into a refusal.
* **Numeric style** (``[1]``, ``[2, 3]``, ``[1][2]``), the convention most RAG
  prompts use, where the number is the passage's 1-based position in the
  context. Converted to tags for checking and back to numbers for output.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

CitationStyle = Literal["tag", "numeric"]

CITATION_RE = re.compile(r"\[chunk:([\w-]+)\]")
_ALT_CITATION_RE = re.compile(r"[\[【［〔]\s*chunk\s*:\s*([\w-]+)\s*[\]】］〕]")
# "[1]", "[1, 2]", "[1][2]", "【3】": digits and separators only, so a bracketed
# word in ordinary prose ("[sic]") is never read as a citation.
_NUMERIC_GROUP_RE = re.compile(r"[\[【［]\s*(\d+(?:\s*[,;]\s*\d+)*)\s*[\]】］]")


def normalize_citations(text: str) -> str:
    """Rewrite every recognised tag-style citation to the canonical ``[chunk:id]``."""
    return _ALT_CITATION_RE.sub(r"[chunk:\1]", text)


def numeric_to_tags(text: str, ids: Sequence[str]) -> str:
    """``[1, 3]`` -> ``[chunk:<ids[0]>] [chunk:<ids[2]>]``.

    A number with no passage behind it (``[9]`` when five were retrieved) is
    kept as a tag naming a passage that does not exist, so the gate treats it
    exactly like any other fabricated citation rather than silently dropping it.
    """

    def repl(m: re.Match[str]) -> str:
        tags: list[str] = []
        for n in re.split(r"\s*[,;]\s*", m.group(1)):
            i = int(n)
            tags.append(f"[chunk:{ids[i - 1]}]" if 1 <= i <= len(ids) else f"[chunk:missing-{i}]")
        return " ".join(tags)

    return _NUMERIC_GROUP_RE.sub(repl, text)


def tags_to_numeric(text: str, ids: Sequence[str]) -> str:
    """``[chunk:<ids[2]>]`` -> ``[3]``. Adjacent tags become ``[1][3]``."""
    position = {pid: i + 1 for i, pid in enumerate(ids)}

    def repl(m: re.Match[str]) -> str:
        return f"[{position[m.group(1)]}]" if m.group(1) in position else ""

    out = CITATION_RE.sub(repl, text)
    return re.sub(r"\]\s+\[", "][", out)


def strip_citations(text: str) -> str:
    """The prose alone, for display surfaces that render citations separately."""
    out = CITATION_RE.sub("", normalize_citations(text))
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return re.sub(r"[ \t]{2,}", " ", out).strip()
