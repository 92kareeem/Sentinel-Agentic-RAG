"""Groq client resilience tests: fail-fast on deterministic errors, retry on
transient ones, and per-model circuit breaker isolation — fully offline via a
fake httpx.Client.
"""

from typing import Any
from unittest.mock import patch

import pytest
from app.llm import groq_client
from app.llm.groq_client import CircuitOpenError, GroqRequestError


class _FakeResponse:
    def __init__(self, status_code: int, json_body: dict[str, Any] | None = None,
                 text: str = "", headers: dict[str, str] | None = None):
        self.status_code = status_code
        self._json = json_body or {}
        self.text = text
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        return self._json


class _FakeClient:
    """Stands in for httpx.Client(timeout=...).

    Returns queued responses in order; once exhausted, repeats the last one
    (useful for "always fails" scenarios spanning multiple chat_completion
    calls, e.g. tripping the circuit breaker's 3-call failure threshold).
    """

    def __init__(self, responses: list[_FakeResponse]):
        self._responses = list(responses)
        self.calls = 0

    def __call__(self, *a, **kw):  # httpx.Client(timeout=15.0) call
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, *a, **kw):
        self.calls += 1
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def _ok(content: str = "hi") -> _FakeResponse:
    return _FakeResponse(200, {"choices": [{"message": {"content": content}}], "usage": {}})


@pytest.fixture(autouse=True)
def _reset_breakers():
    groq_client._breakers.clear()
    yield
    groq_client._breakers.clear()


def test_deterministic_4xx_fails_immediately_without_retry() -> None:
    fake = _FakeClient([_FakeResponse(400, text="bad request")])
    with patch("app.llm.groq_client.httpx.Client", fake):
        with pytest.raises(GroqRequestError):
            groq_client.chat_completion(model="m", messages=[{"role": "user", "content": "x"}])
    assert fake.calls == 1  # no retries burned on a doomed request


def test_429_is_retried_then_succeeds() -> None:
    fake = _FakeClient([_FakeResponse(429, headers={"Retry-After": "0"}), _ok("answer")])
    with patch("app.llm.groq_client.httpx.Client", fake), patch("time.sleep"):
        content, *_ = groq_client.chat_completion(model="m", messages=[])
    assert content == "answer"
    assert fake.calls == 2


def test_circuit_breaker_isolated_per_model() -> None:
    # CircuitBreaker.fail_threshold trips after 3 separate failed
    # chat_completion CALLS, not 3 internal retry attempts within one call —
    # so this must actually invoke chat_completion 3 times.
    fake_a = _FakeClient([_FakeResponse(500)])  # every post() call returns 500
    with patch("app.llm.groq_client.httpx.Client", fake_a), patch("time.sleep"):
        for _ in range(3):
            with pytest.raises(RuntimeError):
                groq_client.chat_completion(model="a", messages=[{"role": "user", "content": "x"}])

    # breaker for "a" is now open -- fails fast, no HTTP call at all
    with pytest.raises(CircuitOpenError):
        groq_client.chat_completion(model="a", messages=[{"role": "user", "content": "x"}])

    # model "b" is untouched -- its own breaker is still closed
    fake_b = _FakeClient([_ok("fine")])
    with patch("app.llm.groq_client.httpx.Client", fake_b):
        content, *_ = groq_client.chat_completion(model="b", messages=[{"role": "user", "content": "x"}])
    assert content == "fine"
