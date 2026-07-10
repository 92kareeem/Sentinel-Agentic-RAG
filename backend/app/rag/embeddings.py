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


_onnx: tuple[Any, Any] | None = None  # (InferenceSession, Tokenizer), lazy


def _get_onnx() -> tuple[Any, Any]:
    global _onnx
    if _onnx is None:
        import onnxruntime
        from tokenizers import Tokenizer

        d = get_settings().onnx_model_dir
        session = onnxruntime.InferenceSession(str(d / "model_quantized.onnx"))
        tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=512)
        _onnx = (session, tokenizer)
    return _onnx


def _embed_onnx(texts: list[str]) -> np.ndarray:
    """Mean-pooled, L2-normalized MiniLM embeddings via onnxruntime.

    Must reproduce sentence-transformers' pipeline exactly (mean pooling over
    the attention mask, then normalize) — verified by scripts/export_onnx.py.
    """
    session, tokenizer = _get_onnx()
    out = []
    for text in texts:
        enc = tokenizer.encode(text)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        hidden = session.run(
            None,
            {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)},
        )[0]  # (1, seq, 384)
        m = mask[..., None].astype(np.float32)
        vec = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        out.append(vec[0])
    vecs = np.asarray(out, dtype=np.float32)
    return vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed to L2-normalized float32 vectors (384-dim).

    Normalized so FAISS inner-product search == cosine similarity. Backend is
    torch (laptop/ingestion) or onnx (Lambda) — identical output space.
    """
    if get_settings().embed_backend == "onnx":
        return _embed_onnx(texts)
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
