"""Embedding model wrapper (lazy singleton).

Role in architecture: the only module that touches sentence-transformers.
The model (~90 MB, 2-5 s load) is loaded once per process on FIRST use and
cached in a module-level global — in Lambda, warm invocations reuse it for
free; /healthz never pays for it.
"""

import threading
from typing import Any

import numpy as np

from app.config import get_settings

# Guards first-use construction of every lazy singleton below.
#
# `if _x is None: _x = build()` is not safe when two threads arrive together:
# both see None and both build. For the torch model that is not merely wasteful
# — concurrent SentenceTransformer construction fails outright with
# "NotImplementedError: Cannot copy out of meta tensor; no data!", because two
# threads materialize the same meta-device parameters at once.
#
# That is reachable in normal operation, not just in tests: FastAPI runs `def`
# endpoints in a threadpool, so two uploads arriving together in one process
# race here on the very first request — the worst possible moment, since a cold
# process is exactly when nothing is cached yet.
#
# One lock for all three singletons: initialization happens once per process,
# so contention is irrelevant, and only one embedding backend is ever active.
_init_lock = threading.Lock()

# Guards USE of the torch model's tokenizer, which is a different problem from
# guarding its construction.
#
# SentenceTransformer exposes one HuggingFace fast tokenizer, and this module
# calls it two ways: token_offsets() asks for offset mappings with
# truncation=False, while encode() tokenizes for the model with truncation on.
# The HF wrapper applies those settings by MUTATING the underlying Rust
# tokenizer before each call, so a mutation racing an in-flight encode fails
# with "RuntimeError: Already borrowed" — Rust refusing a mutable borrow while
# a shared one is live.
#
# Reachable in normal operation: chunking calls token_offsets() OUTSIDE the
# index publication lock, so two documents ingesting at once in one process
# hit exactly this pair of calls concurrently. It surfaced as a test that
# failed roughly one run in five with an error that names nothing in this
# codebase.
#
# One lock over both uses rather than a second tokenizer instance: it needs no
# extra model load, and the serialised sections are short next to the model
# forward pass, which is already serialised during publication anyway.
#
# The ONNX path does not need this. It builds two separate tokenizers.Tokenizer
# objects and configures each once, under _init_lock, before publishing it —
# after that every call is read-only.
_torch_tokenizer_lock = threading.Lock()

_model: Any = None  # sentence_transformers.SentenceTransformer, loaded lazily


def get_model() -> Any:
    """Double-checked locking: the fast path stays lock-free once loaded.

    The unlocked read is safe because assignment to a module global is atomic
    under the GIL — a reader sees either None or a fully-constructed model,
    never a half-built one.
    """
    global _model
    if _model is None:
        with _init_lock:
            if _model is None:  # re-check: another thread may have built it
                from sentence_transformers import SentenceTransformer  # heavy, kept lazy

                _model = SentenceTransformer(get_settings().embed_model_name, device="cpu")
    return _model


_onnx: tuple[Any, Any] | None = None  # (InferenceSession, Tokenizer), lazy


def _get_onnx() -> tuple[Any, Any]:
    global _onnx
    if _onnx is None:
        with _init_lock:
            if _onnx is None:
                import onnxruntime
                from tokenizers import Tokenizer

                d = get_settings().onnx_model_dir
                session = onnxruntime.InferenceSession(str(d / "model_quantized.onnx"))
                tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
                tokenizer.enable_truncation(max_length=256)  # all-MiniLM-L6-v2 real max
                _onnx = (session, tokenizer)
    return _onnx


def _embed_onnx(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Mean-pooled, L2-normalized MiniLM embeddings via onnxruntime.

    Batched with padding so ingesting a whole document is fast enough to run
    inside a single Lambda invocation. Reproduces sentence-transformers'
    pipeline exactly (mean pooling over the mask, then normalize) — verified
    by scripts/export_onnx.py.
    """
    session, tokenizer = _get_onnx()
    out: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        encs = [tokenizer.encode(t) for t in texts[start : start + batch_size]]
        maxlen = max((len(e.ids) for e in encs), default=1)
        ids = np.zeros((len(encs), maxlen), dtype=np.int64)
        mask = np.zeros((len(encs), maxlen), dtype=np.int64)
        for j, e in enumerate(encs):
            ids[j, : len(e.ids)] = e.ids
            mask[j, : len(e.attention_mask)] = e.attention_mask
        hidden = session.run(
            None,
            {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)},
        )[0]  # (batch, seq, 384)
        m = mask[..., None].astype(np.float32)
        out.append((hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None))
    vecs = np.vstack(out).astype(np.float32)
    return vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed to L2-normalized float32 vectors (384-dim).

    Normalized so FAISS inner-product search == cosine similarity. Backend is
    torch (laptop/ingestion) or onnx (Lambda) — identical output space.
    """
    if get_settings().embed_backend == "onnx":
        return _embed_onnx(texts)
    model = get_model()  # loaded outside the lock; only the call is serialised
    with _torch_tokenizer_lock:
        vecs = model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
    return np.asarray(vecs, dtype=np.float32)


_offsets_tok: Any = None  # tokenizers.Tokenizer for chunking (no truncation)


def _onnx_token_offsets(text: str) -> list[tuple[int, int]]:
    global _offsets_tok
    if _offsets_tok is None:
        with _init_lock:
            if _offsets_tok is None:
                from tokenizers import Tokenizer

                tok = Tokenizer.from_file(
                    str(get_settings().onnx_model_dir / "tokenizer.json")
                )
                # tokenizer.json ships with truncation (~128 tokens) enabled — that
                # must NOT apply when computing chunk offsets, or the chunker only
                # ever sees the first 128 tokens of each section and silently drops
                # the rest.
                tok.no_truncation()
                # Published only after no_truncation(): assigning the global first
                # would let another thread grab a tokenizer that still truncates.
                _offsets_tok = tok
    return [(int(a), int(b)) for a, b in _offsets_tok.encode(text).offsets if b > a]


def token_offsets(text: str) -> list[tuple[int, int]]:
    """(char_start, char_end) per token from the MiniLM tokenizer.

    Injected into the chunker so chunk sizes match what the embedder actually
    sees. Dispatches on backend so the same chunker runs under torch (laptop
    ingestion) and onnx (in-Lambda upload processing) — same WordPiece
    tokenizer, so offsets are identical either way.
    """
    if get_settings().embed_backend == "onnx":
        return _onnx_token_offsets(text)
    tokenizer = get_model().tokenizer
    with _torch_tokenizer_lock:
        enc = tokenizer(
            text, return_offsets_mapping=True, add_special_tokens=False, truncation=False
        )
    return [(int(a), int(b)) for a, b in enc["offset_mapping"] if b > a]
