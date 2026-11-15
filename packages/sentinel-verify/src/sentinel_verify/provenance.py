"""A citation hash anyone can recompute.

"This quote is characters 120 to 410 of the file whose SHA-256 is X" is a
claim. The hash makes it checkable without trusting the server that made it:
``sha256(document_sha256 | char_start | char_end | text)``. A client holding
those four things, in any language, can prove the quote is that exact span of
that exact file. The recipe is deliberately trivial so that stays true.
"""

from __future__ import annotations

import hashlib

HASH_RECIPE = "sha256(document_sha256 + '|' + char_start + '|' + char_end + '|' + text)"


def provenance_hash(document_sha256: str, char_start: int, char_end: int, text: str) -> str:
    payload = f"{document_sha256}|{char_start}|{char_end}|{text}".encode()
    return hashlib.sha256(payload).hexdigest()


def document_sha256(data: bytes) -> str:
    """The checksum of a source file, as the recipe expects it."""
    return hashlib.sha256(data).hexdigest()


def check_provenance(
    *, document_sha256: str, char_start: int, char_end: int, text: str, expected: str
) -> bool:
    """Does this passage's hash match the span it claims? Constant-time compare."""
    actual = provenance_hash(document_sha256, char_start, char_end, text)
    return hashlib.sha256(actual.encode()).digest() == hashlib.sha256(expected.encode()).digest()
