"""Regex PII scrub, applied BEFORE the query is logged or sent to Groq.

Role in architecture: deterministic, fast, free, and crucially never itself
ships data to a third party (an LLM-based detector would). Replacements are
typed placeholders so logs stay debuggable without leaking.
"""

import re

_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("phone", re.compile(r"(?<![\d-])\+?\d[\d\s().-]{7,14}\d(?![\d-])")),
]


def scrub(text: str) -> str:
    for label, pattern in _RULES:
        text = pattern.sub(f"[PII:{label}]", text)
    return text
