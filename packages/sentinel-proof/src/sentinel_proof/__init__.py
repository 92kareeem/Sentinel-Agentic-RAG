"""sentinel-proof: a deterministic grounding gate for RAG answers.

    from sentinel_proof import verify, passages_from_texts

    passages = passages_from_texts(["Customers may request a refund within 30 days."])
    v = verify("Refunds are available for 60 days [1].", passages, citation_style="numeric")
    v.ok            # False: the only claim's number is not in the evidence
    v.safe_text()   # refusal prose generated from the reason code, never the draft

Pure Python, no dependencies, no model calls. See the README for the method.
"""

from .citations import CitationStyle, normalize_citations, strip_citations
from .grounding import (
    MAX_STRIPPED_RATIO,
    MIN_LEXICAL_OVERLAP,
    Claim,
    ClaimStatus,
    Verdict,
    verify,
)
from .passages import Passage, from_langchain, from_llamaindex, passages_from_texts
from .provenance import HASH_RECIPE, check_provenance, document_sha256, provenance_hash
from .refusal import REFUSAL_SENTINEL, RefusalReason, is_refusal, refusal_text

__version__ = "0.1.0"

__all__ = [
    "HASH_RECIPE",
    "MAX_STRIPPED_RATIO",
    "MIN_LEXICAL_OVERLAP",
    "REFUSAL_SENTINEL",
    "CitationStyle",
    "Claim",
    "ClaimStatus",
    "Passage",
    "RefusalReason",
    "Verdict",
    "__version__",
    "check_provenance",
    "document_sha256",
    "from_langchain",
    "from_llamaindex",
    "is_refusal",
    "normalize_citations",
    "passages_from_texts",
    "provenance_hash",
    "refusal_text",
    "strip_citations",
    "verify",
]
