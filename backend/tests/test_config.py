"""Secret resolution in get_settings().

The Groq key reaches the app one of two ways: GROQ_API_KEY directly (local
.env), or GROQ_API_KEY_SSM_PARAM naming an SSM SecureString (AWS). These tests
pin the precedence and the failure behaviour without touching real AWS.
"""

from __future__ import annotations

from typing import Any

import pytest
from app import config


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY_SSM_PARAM", raising=False)
    # .env in the repo root must not leak a real key into these tests
    monkeypatch.setattr(
        config.Settings, "model_config", {**config.Settings.model_config, "env_file": None}
    )
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def test_ssm_parameter_is_fetched_when_no_direct_key(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_fetch(name: str, region: str) -> str:
        calls.append((name, region))
        return "gsk_from_ssm"

    monkeypatch.setattr(config, "_fetch_ssm_secret", fake_fetch)
    monkeypatch.setenv("GROQ_API_KEY_SSM_PARAM", "/sentinel/groq-api-key")

    assert config.get_settings().groq_api_key == "gsk_from_ssm"
    assert calls == [("/sentinel/groq-api-key", "ap-south-1")]


def test_ssm_is_read_once_per_process_not_per_call(monkeypatch) -> None:
    """On Lambda this is once per cold start. Per-request SSM calls would add
    latency to every query and burn SSM API quota for no benefit."""
    calls: list[str] = []
    monkeypatch.setattr(config, "_fetch_ssm_secret", lambda n, r: calls.append(n) or "k")
    monkeypatch.setenv("GROQ_API_KEY_SSM_PARAM", "/sentinel/groq-api-key")

    for _ in range(5):
        config.get_settings()
    assert len(calls) == 1


def test_direct_key_wins_and_ssm_is_never_called(monkeypatch) -> None:
    """Local development must work with a .env file and no AWS credentials."""

    def must_not_run(name: str, region: str) -> str:
        raise AssertionError("SSM should not be called when GROQ_API_KEY is set")

    monkeypatch.setattr(config, "_fetch_ssm_secret", must_not_run)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_local")
    monkeypatch.setenv("GROQ_API_KEY_SSM_PARAM", "/sentinel/groq-api-key")

    assert config.get_settings().groq_api_key == "gsk_local"


def test_neither_set_leaves_key_empty_without_calling_aws(monkeypatch) -> None:
    monkeypatch.setattr(
        config, "_fetch_ssm_secret", lambda n, r: pytest.fail("unexpected SSM call")
    )
    assert config.get_settings().groq_api_key == ""


def test_unreadable_parameter_fails_loudly_without_leaking_a_value(monkeypatch) -> None:
    """A broken secret must stop startup, not surface later as a Groq 401."""
    import sys
    import types

    from botocore.exceptions import ClientError

    class FakeSSM:
        def get_parameter(self, **kwargs: Any) -> dict[str, Any]:
            raise ClientError(
                {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
                "GetParameter",
            )

    fake_boto3 = types.SimpleNamespace(client=lambda service, region_name: FakeSSM())
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setenv("GROQ_API_KEY_SSM_PARAM", "/sentinel/groq-api-key")

    with pytest.raises(RuntimeError, match="/sentinel/groq-api-key"):
        config.get_settings()


def test_empty_parameter_is_rejected(monkeypatch) -> None:
    import sys
    import types

    class FakeSSM:
        def get_parameter(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["WithDecryption"] is True  # SecureString must be decrypted
            return {"Parameter": {"Value": "   "}}

    monkeypatch.setitem(
        sys.modules, "boto3", types.SimpleNamespace(client=lambda s, region_name: FakeSSM())
    )
    monkeypatch.setenv("GROQ_API_KEY_SSM_PARAM", "/sentinel/groq-api-key")

    with pytest.raises(RuntimeError, match="empty"):
        config.get_settings()
