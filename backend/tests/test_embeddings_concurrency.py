"""Thread-safety of the lazily-initialized embedding singletons.

Found by a flaky concurrency test that failed ~4 runs in 5:

    NotImplementedError: Cannot copy out of meta tensor; no data!

Four threads ingesting at once each called token_offsets() during chunking —
which happens outside the index publication lock — and raced while building the
model. This is reachable in production, not only in tests: FastAPI runs `def`
endpoints in a threadpool, so two uploads arriving together in a cold process
hit it on the very first request.

These tests drive the locking directly with fake, deliberately-slow
constructors rather than real models: the real failure depends on torch
internals and reproduces only intermittently, whereas "the factory runs exactly
once" is the invariant that actually matters and can be asserted every time.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from typing import Any

import pytest
from app.rag import embeddings


def _run_concurrently(fn: Any, n: int = 8) -> list[Exception]:
    errors: list[Exception] = []
    barrier = threading.Barrier(n)

    def worker() -> None:
        try:
            barrier.wait()  # maximize overlap on the first-use path
            fn()
        except Exception as exc:  # noqa: BLE001 — surfaced via `errors`
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_torch_model_is_constructed_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two concurrent constructions of SentenceTransformer do not merely waste
    memory — they raise, because both threads materialize the same meta-device
    parameters at once."""
    built: list[str] = []

    class FakeModel:
        def __init__(self, name: str, device: str | None = None) -> None:
            built.append(name)
            time.sleep(0.05)  # widen the window a real load would have

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeModel)
    )
    monkeypatch.setattr(embeddings, "_model", None)

    errors = _run_concurrently(embeddings.get_model)

    assert not errors, errors
    assert built == ["sentence-transformers/all-MiniLM-L6-v2"], built


def test_offsets_tokenizer_is_constructed_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[int] = []

    class FakeTokenizer:
        def __init__(self) -> None:
            self.truncating = True

        @staticmethod
        def from_file(path: str) -> FakeTokenizer:
            built.append(1)
            time.sleep(0.05)
            return FakeTokenizer()

        def no_truncation(self) -> None:
            self.truncating = False

        def encode(self, text: str) -> Any:
            return types.SimpleNamespace(offsets=[(0, 1)])

    monkeypatch.setenv("EMBED_BACKEND", "onnx")
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setitem(sys.modules, "tokenizers", types.SimpleNamespace(Tokenizer=FakeTokenizer))
    monkeypatch.setattr(embeddings, "_offsets_tok", None)

    errors = _run_concurrently(lambda: embeddings.token_offsets("hello world"))
    get_settings.cache_clear()

    assert not errors, errors
    assert len(built) == 1, f"tokenizer built {len(built)} times"


def test_offsets_tokenizer_is_never_visible_while_it_still_truncates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordering bug, separate from the race: the global used to be assigned
    BEFORE no_truncation() was applied, so another thread could grab a
    tokenizer that still truncated at ~128 tokens — silently dropping the tail
    of every section from chunking rather than failing."""
    observed_truncating: list[bool] = []

    class FakeTokenizer:
        def __init__(self) -> None:
            self.truncating = True

        @staticmethod
        def from_file(path: str) -> FakeTokenizer:
            return FakeTokenizer()

        def no_truncation(self) -> None:
            time.sleep(0.05)  # a window for another thread to observe the global
            self.truncating = False

        def encode(self, text: str) -> Any:
            observed_truncating.append(self.truncating)
            return types.SimpleNamespace(offsets=[(0, 1)])

    monkeypatch.setenv("EMBED_BACKEND", "onnx")
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setitem(sys.modules, "tokenizers", types.SimpleNamespace(Tokenizer=FakeTokenizer))
    monkeypatch.setattr(embeddings, "_offsets_tok", None)

    errors = _run_concurrently(lambda: embeddings.token_offsets("hello world"))
    get_settings.cache_clear()

    assert not errors, errors
    assert observed_truncating, "no thread reached encode()"
    assert not any(observed_truncating), "a thread used a still-truncating tokenizer"


def test_offsets_and_encode_never_touch_the_shared_tokenizer_at_once() -> None:
    """The torch backend's two tokenizer callers must not overlap.

    SentenceTransformer exposes ONE HuggingFace fast tokenizer, and this module
    calls it two ways: token_offsets() wants offset mappings with
    truncation=False, encode() tokenizes for the model with truncation on. The
    HF wrapper applies those settings by mutating the underlying Rust
    tokenizer, so a mutation racing an in-flight encode raises

        RuntimeError: Already borrowed

    Reachable in normal operation, not only in tests: chunking calls
    token_offsets() OUTSIDE the index publication lock, so two documents
    ingesting at once in one process hit exactly this pair concurrently. It
    showed up as a suite that failed about one run in five with an error
    naming nothing in this codebase.

    Driven with a fake tokenizer that RECORDS overlap rather than with the
    real one: the genuine failure depends on Rust borrow timing and reproduces
    intermittently, whereas "these two callers are never inside the tokenizer
    together" is the invariant that actually matters and can be asserted every
    run.
    """
    overlapping: list[str] = []
    inside = threading.Lock()
    occupants: list[str] = []

    def enter(who: str) -> None:
        with inside:
            occupants.append(who)
            if len(occupants) > 1:
                overlapping.append(",".join(sorted(occupants)))
        time.sleep(0.01)  # hold the tokenizer long enough for a race to show
        with inside:
            occupants.remove(who)

    class FakeTokenizer:
        def __call__(self, text: str, **kw: Any) -> dict[str, Any]:
            enter("offsets")  # mirrors token_offsets(): mutates truncation
            return {"offset_mapping": [(0, 5)]}

    class FakeModel:
        tokenizer = FakeTokenizer()

        def encode(self, texts: list[str], **kw: Any) -> Any:
            import numpy as np

            enter("encode")
            return np.zeros((len(texts), 384), dtype="float32")

    embeddings._model = FakeModel()
    try:
        errors = _run_concurrently(
            lambda: (
                embeddings.token_offsets("some section text"),
                embeddings.embed_texts(["some section text"]),
            ),
            n=6,
        )
    finally:
        embeddings._model = None

    assert not errors, errors
    assert not overlapping, f"tokenizer used concurrently by {set(overlapping)}"
