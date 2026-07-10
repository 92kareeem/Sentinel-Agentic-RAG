"""Embedding model wrapper (lazy singleton).

Role in architecture: the only module that touches sentence-transformers.
The model (~90 MB, 2-5 s load) is loaded once per process on FIRST use and
cached in a module-level global — in Lambda, warm invocations reuse it for
free; /healthz never pays for it.
"""

from typing import Any

import numpy as np

from app.config import get_settings

_model: Any = None  # sentence_transformers.SentenceTransformer, loaded lazily


def get_model() -> Any:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # heavy import kept lazy

        _model = SentenceTransformer(get_settings().embed_model_name, device="cpu")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed to L2-normalized float32 vectors (384-dim).

    Normalized so FAISS inner-product search == cosine similarity.
    """
    vecs = get_model().encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    return np.asarray(vecs, dtype=np.float32)


def token_offsets(text: str) -> list[tuple[int, int]]:
    """(char_start, char_end) per token from the MiniLM fast tokenizer.

    Injected into the chunker so chunk sizes match what the embedder
    actually sees (512-token model limit).
    """
    enc = get_model().tokenizer(
        text, return_offsets_mapping=True, add_special_tokens=False, truncation=False
    )
    return [(int(a), int(b)) for a, b in enc["offset_mapping"] if b > a]
