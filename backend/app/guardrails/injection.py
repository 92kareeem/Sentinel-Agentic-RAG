"""Heuristic prompt-injection screen.

Role in architecture: noise reduction + metrics, NOT a security boundary.
The real boundary is architectural: the model only sees retrieved chunks and
has no tools to abuse. These patterns catch the obvious attempts (-> 422
with a reason code we log and count as InjectionBlocked).
"""

import re

from fastapi import HTTPException

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "override_instructions",
        re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions", re.I),
    ),
    (
        "system_prompt_probe",
        re.compile(r"(system\s+prompt|you\s+are\s+now|new\s+instructions:)", re.I),
    ),
    ("role_tag", re.compile(r"<\s*/?\s*(system|assistant|user)\s*>|\[/?(INST|SYS)\]", re.I)),
    ("base64_blob", re.compile(r"[A-Za-z0-9+/=]{200,}")),
    ("zero_width", re.compile(r"[​‌‍⁠﻿]")),
    ("control_chars", re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]{3,}")),
]


def screen_query(query: str) -> str:
    """Return the query unchanged, or raise 422 with the matched reason code."""
    for reason, pattern in _PATTERNS:
        if pattern.search(query):
            raise HTTPException(
                status_code=422,
                detail=f"query flagged by injection screen: {reason}",
            )
    return query
