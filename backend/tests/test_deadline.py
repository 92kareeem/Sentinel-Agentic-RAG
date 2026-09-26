"""The request deadline has to bound provider calls, not just the gaps between them.

Found by running the system end to end against live Groq: a request configured
with DEADLINE_SECONDS=45 took 285 seconds and then refused. check_budget() runs
BETWEEN graph nodes, so a node already inside a provider call could not be
interrupted — and Groq answers a free-tier token-per-minute limit with a
Retry-After measured in tens of seconds, retried three times, across several
nodes.

On a laptop that is a slow answer. In Lambda it is a function timeout and a
502 with no refusal at all, which is the failure this system is specifically
supposed not to have.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest
from app.agents.budget import llm_call
from app.llm import groq_client


class _Resp:
    """A 429 carrying the kind of Retry-After Groq's free tier actually sends."""

    status_code = 429
    text = "rate limited"

    def __init__(self, retry_after: str = "60") -> None:
        self.headers = {"Retry-After": retry_after}

    def json(self) -> dict[str, Any]:  # pragma: no cover - never reached on 429
        return {}


@pytest.fixture(autouse=True)
def _fresh_breaker() -> Any:
    groq_client._breakers.clear()
    yield
    groq_client._breakers.clear()


def _always_rate_limited(monkeypatch: pytest.MonkeyPatch, retry_after: str = "60") -> list[float]:
    """Make every call 429, and record any sleep instead of performing it."""
    slept: list[float] = []

    class _Client:
        def __init__(self, **kw: Any) -> None:
            self.timeout = kw.get("timeout")

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def post(self, *_: Any, **__: Any) -> _Resp:
            return _Resp(retry_after)

    monkeypatch.setattr(groq_client.httpx, "Client", _Client)
    monkeypatch.setattr(groq_client.time, "sleep", lambda s: slept.append(s))
    return slept


def test_a_retry_that_would_outlast_the_deadline_is_not_attempted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE regression test.

    Sleeping past the deadline is strictly worse than failing now: the answer
    is discarded either way, and the caller spends the difference holding a
    connection open. Refusing early leaves time to say so.
    """
    slept = _always_rate_limited(monkeypatch, retry_after="60")

    with pytest.raises(groq_client.GroqDeadlineError):
        groq_client.chat_completion(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            deadline_ts=time.monotonic() + 5,  # 5s left, Groq wants 60
        )

    assert slept == [], f"slept past the deadline: {slept}"


def test_a_retry_that_fits_inside_the_deadline_still_happens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bound must not turn into 'never retry' — a transient 429 that we can
    afford to wait out is exactly what the retry loop is for."""
    slept = _always_rate_limited(monkeypatch, retry_after="1")

    with pytest.raises((RuntimeError, groq_client.GroqDeadlineError)):
        groq_client.chat_completion(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            deadline_ts=time.monotonic() + 300,
        )

    assert slept, "an affordable retry was skipped"


def test_a_call_started_after_the_deadline_is_refused_without_touching_the_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    class _Client:
        def __init__(self, **kw: Any) -> None:
            pass

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def post(self, *_: Any, **__: Any) -> None:
            nonlocal called
            called = True

    monkeypatch.setattr(groq_client.httpx, "Client", _Client)

    with pytest.raises(groq_client.GroqDeadlineError):
        groq_client.chat_completion(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            deadline_ts=time.monotonic() - 1,
        )

    assert not called, "spent a provider call on a request that had already expired"


def test_the_http_timeout_never_exceeds_what_the_request_can_afford(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One hung connection must not consume a budget three nodes still share."""
    seen: list[float] = []

    class _Client:
        def __init__(self, **kw: Any) -> None:
            seen.append(kw["timeout"])

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def post(self, *_: Any, **__: Any) -> None:
            raise httpx.ConnectError("boom")

    monkeypatch.setattr(groq_client.httpx, "Client", _Client)
    monkeypatch.setattr(groq_client.time, "sleep", lambda s: None)

    with pytest.raises((RuntimeError, groq_client.GroqDeadlineError)):
        groq_client.chat_completion(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            deadline_ts=time.monotonic() + 3,
        )

    assert seen and all(t <= 3.01 for t in seen), seen


def test_an_unbounded_call_keeps_the_fixed_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """deadline_ts=None (ingestion, scripts) must still cap a single attempt."""
    seen: list[float] = []

    class _Client:
        def __init__(self, **kw: Any) -> None:
            seen.append(kw["timeout"])

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def post(self, *_: Any, **__: Any) -> None:
            raise httpx.ConnectError("boom")

    monkeypatch.setattr(groq_client.httpx, "Client", _Client)
    monkeypatch.setattr(groq_client.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError):
        groq_client.chat_completion(model="m", messages=[{"role": "user", "content": "hi"}])

    assert seen == [groq_client.MAX_CALL_SECONDS] * 3, seen


# ------------------------------------------------- what the graph does with it


def test_the_graph_turns_a_deadline_into_a_retryable_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not a 503. The corpus may well contain the answer — the user should be
    told to retry, not told their documents are missing something."""

    def boom(**_: Any) -> None:
        raise groq_client.GroqDeadlineError("out of time")

    monkeypatch.setattr(groq_client, "chat_completion", boom)

    state: dict[str, Any] = {"deadline_ts": time.monotonic() + 10, "status": "running"}
    assert llm_call(state, model="m", messages=[]) is None
    assert state["status"] == "refused"


def test_every_llm_call_in_the_graph_goes_through_the_bounded_helper() -> None:
    """A bound that one call site can forget is not a bound.

    groq_client.chat_completion must be reached only via budget.llm_call, which
    is the single place the deadline is attached.
    """
    import pathlib

    app = pathlib.Path(__file__).resolve().parents[1] / "app"
    # groq_client.py declares it; budget.py is the one place allowed to call it.
    allowed = {"llm/groq_client.py", "agents/budget.py"}
    offenders = [
        rel
        for path in app.rglob("*.py")
        if (rel := path.relative_to(app).as_posix()) not in allowed
        and "chat_completion(" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"unbounded LLM call sites: {offenders}"
