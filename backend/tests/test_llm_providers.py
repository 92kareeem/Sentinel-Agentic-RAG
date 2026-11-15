"""Every provider adapter speaks its own wire format over the one shared core.

Offline, through a fake httpx.Client: these tests pin the URL, the auth
header and the body shape each vendor expects, and that the response is
normalised into the same (content, tokens_in, tokens_out) the graph consumes.
A provider that quietly sent the wrong header would fail every request in
production with a 401 that looks like a key problem; here it fails a test.
"""

from __future__ import annotations

from typing import Any

import pytest
from app import config
from app.llm import anthropic_client, base, groq_client, openai_client, pricing, registry
from app.llm.base import LLMRequestError


class _Resp:
    def __init__(self, status: int, body: dict[str, Any], headers: dict[str, str] | None = None):
        self.status_code = status
        self._body = body
        self.text = str(body)
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        return self._body


class _Capture:
    """Fake httpx.Client that records the request and replays queued responses."""

    def __init__(self, responses: list[_Resp]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kw: Any) -> _Capture:
        return self

    def __enter__(self) -> _Capture:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str]) -> _Resp:
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def _openai_ok(content: str = "hi") -> _Resp:
    return _Resp(
        200,
        {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        },
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(
        config.Settings, "model_config", {**config.Settings.model_config, "env_file": None}
    )
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk_test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak_test")
    monkeypatch.setenv("ANTHROPIC_MODEL_SIMPLE", "model-s")
    monkeypatch.setenv("ANTHROPIC_MODEL_COMPLEX", "model-c")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://acme.openai.azure.com")
    config.get_settings.cache_clear()
    pricing.reset_cache()
    base.BREAKERS.clear()
    yield
    config.get_settings.cache_clear()
    pricing.reset_cache()
    base.BREAKERS.clear()


def _use(monkeypatch: pytest.MonkeyPatch, fake: _Capture) -> None:
    monkeypatch.setattr("app.llm.http_chat.httpx.Client", fake)


# ----------------------------------------------------------------- adapters


def test_openai_adapter_uses_bearer_auth_and_chat_completions(monkeypatch) -> None:
    fake = _Capture([_openai_ok("answer")])
    _use(monkeypatch, fake)
    out = openai_client.chat_completion("gpt-4o-mini", [{"role": "user", "content": "q"}])
    assert out == ("answer", 11, 7)
    call = fake.calls[0]
    assert call["url"] == "https://api.openai.com/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer sk_test"
    assert call["json"]["model"] == "gpt-4o-mini"


def test_openai_json_mode_requests_a_json_object(monkeypatch) -> None:
    fake = _Capture([_openai_ok("{}")])
    _use(monkeypatch, fake)
    openai_client.chat_completion("m", [], json_mode=True)
    assert fake.calls[0]["json"]["response_format"] == {"type": "json_object"}


def test_azure_adapter_routes_by_deployment_with_api_key_header(monkeypatch) -> None:
    """Azure differs in exactly three ways: path carries the deployment, the
    api-version is a query parameter, and auth is `api-key`, not Bearer."""
    fake = _Capture([_openai_ok()])
    _use(monkeypatch, fake)
    openai_client.chat_completion("my-deploy", [], flavour="azure")
    call = fake.calls[0]
    assert call["url"] == (
        "https://acme.openai.azure.com/openai/deployments/my-deploy/chat/completions"
        "?api-version=2024-10-21"
    )
    assert call["headers"]["api-key"] == "sk_test"
    assert "Authorization" not in call["headers"]
    assert "model" not in call["json"]


def test_local_adapter_targets_the_configured_server(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://ollama:11434/v1/")
    config.get_settings.cache_clear()
    fake = _Capture([_openai_ok()])
    _use(monkeypatch, fake)
    openai_client.chat_completion("llama3.1:8b", [], flavour="local")
    assert fake.calls[0]["url"] == "http://ollama:11434/v1/chat/completions"


def test_anthropic_adapter_maps_system_prompt_blocks_and_usage(monkeypatch) -> None:
    fake = _Capture(
        [
            _Resp(
                200,
                {
                    "content": [
                        {"type": "text", "text": "Refunds take "},
                        {"type": "text", "text": "30 days."},
                    ],
                    "usage": {"input_tokens": 40, "output_tokens": 9},
                    "stop_reason": "end_turn",
                },
            )
        ]
    )
    _use(monkeypatch, fake)
    out = anthropic_client.chat_completion(
        "model-s",
        [{"role": "system", "content": "Be strict."}, {"role": "user", "content": "q"}],
        max_tokens=200,
    )
    assert out == ("Refunds take 30 days.", 40, 9)
    call = fake.calls[0]
    assert call["url"] == "https://api.anthropic.com/v1/messages"
    assert call["headers"]["x-api-key"] == "ak_test"
    assert call["headers"]["anthropic-version"]
    assert call["json"]["system"] == "Be strict."
    assert call["json"]["messages"] == [{"role": "user", "content": "q"}]
    assert call["json"]["max_tokens"] == 200
    assert "temperature" not in call["json"]


def test_anthropic_json_mode_strips_a_code_fence_and_validates(monkeypatch) -> None:
    fenced = '```json\n{"faithfulness": 1.0, "relevance": 1.0}\n```'
    fake = _Capture(
        [_Resp(200, {"content": [{"type": "text", "text": fenced}], "usage": {}})]
    )
    _use(monkeypatch, fake)
    content, *_ = anthropic_client.chat_completion("model-s", [], json_mode=True)
    assert content == '{"faithfulness": 1.0, "relevance": 1.0}'
    assert "JSON object" in fake.calls[0]["json"]["system"]


def test_anthropic_refusal_stop_reason_is_a_deterministic_error(monkeypatch) -> None:
    """A provider-side safety decline cannot be retried into a different
    answer; retrying would spend the request's deadline for nothing."""
    fake = _Capture([_Resp(200, {"content": [], "usage": {}, "stop_reason": "refusal"})])
    _use(monkeypatch, fake)
    with pytest.raises(RuntimeError):
        anthropic_client.chat_completion("model-s", [])


def test_anthropic_requires_a_configured_model() -> None:
    with pytest.raises(ValueError, match="ANTHROPIC_MODEL"):
        anthropic_client.chat_completion("", [])


def test_deterministic_4xx_is_not_retried_on_any_adapter(monkeypatch) -> None:
    fake = _Capture([_Resp(401, {"error": "bad key"})])
    _use(monkeypatch, fake)
    with pytest.raises(LLMRequestError):
        openai_client.chat_completion("m", [])
    assert len(fake.calls) == 1


def test_groq_aliases_still_resolve_to_the_shared_errors() -> None:
    assert groq_client.GroqDeadlineError is base.LLMDeadlineError
    assert groq_client.GroqRequestError is base.LLMRequestError
    assert groq_client._breakers is base.BREAKERS


# ----------------------------------------------------------------- registry


def test_registry_dispatches_on_the_configured_provider(monkeypatch) -> None:
    seen: list[str] = []

    def fake_openai(model: str, messages: list, *, flavour: str, **kw: Any) -> tuple[str, int, int]:
        seen.append(flavour)
        return "ok", 1, 2

    monkeypatch.setattr(openai_client, "chat_completion", fake_openai)
    for provider in ("openai", "azure", "local"):
        monkeypatch.setenv("LLM_PROVIDER", provider)
        if provider == "azure":
            monkeypatch.setenv("AZURE_DEPLOYMENT_SIMPLE", "dep")
        config.get_settings.cache_clear()
        c = registry.complete("m", [])
        assert c.provider == provider and c.as_tuple == ("ok", 1, 2)
    assert seen == ["openai", "azure", "local"]


def test_registry_prices_the_call(monkeypatch) -> None:
    monkeypatch.setattr(groq_client, "chat_completion", lambda **_: ("x", 1_000_000, 1_000_000))
    c = registry.complete("openai/gpt-oss-20b", [])
    assert c.provider == "groq"
    assert c.cost_usd == pytest.approx(0.10 + 0.50)


def test_unknown_model_costs_zero_rather_than_a_guess(monkeypatch) -> None:
    monkeypatch.setattr(groq_client, "chat_completion", lambda **_: ("x", 500, 500))
    assert registry.complete("some/new-model", []).cost_usd == 0.0


def test_pricing_overrides_win_over_the_builtin_table(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PRICING_OVERRIDES", "groq/openai/gpt-oss-20b=1:2, bogus")
    config.get_settings.cache_clear()
    pricing.reset_cache()
    assert pricing.price_for("groq", "openai/gpt-oss-20b") == (1.0, 2.0)
    assert pricing.estimate_cost("groq", "openai/gpt-oss-20b", 1_000_000, 0) == 1.0


def test_local_inference_is_free_by_definition() -> None:
    assert pricing.estimate_cost("local", "llama3.1:8b", 10_000, 10_000) == 0.0


# ------------------------------------------------------------- model choice


def test_model_properties_follow_the_provider(monkeypatch) -> None:
    assert config.get_settings().model_simple == "openai/gpt-oss-20b"
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    config.get_settings.cache_clear()
    s = config.get_settings()
    assert (s.model_simple, s.model_complex) == ("model-s", "model-c")


# ------------------------------------------------------------ storage toggle


def test_storage_backend_local_derives_the_legacy_flags(monkeypatch) -> None:
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_MODE", "false")  # ignored: STORAGE_BACKEND wins
    config.get_settings.cache_clear()
    s = config.get_settings()
    assert s.local_mode is True and s.use_s3_index is False


def test_storage_backend_aws_requires_a_bucket(monkeypatch) -> None:
    monkeypatch.setenv("STORAGE_BACKEND", "aws")
    config.get_settings.cache_clear()
    with pytest.raises(ValueError, match="S3_BUCKET_DOCS"):
        config.get_settings()
    monkeypatch.setenv("S3_BUCKET_DOCS", "b")
    config.get_settings.cache_clear()
    s = config.get_settings()
    assert s.local_mode is False and s.use_s3_index is True


def test_legacy_local_mode_flag_still_sets_the_backend(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_MODE", "true")
    config.get_settings.cache_clear()
    assert config.get_settings().storage_backend == "local"


def test_legacy_local_mode_false_does_not_demand_a_bucket(monkeypatch) -> None:
    # DynamoDB-only tooling and tests run with LOCAL_MODE=false and no bucket;
    # only the explicit STORAGE_BACKEND=aws switch validates the bucket up front.
    monkeypatch.delenv("STORAGE_BACKEND", raising=False)
    monkeypatch.setenv("LOCAL_MODE", "false")
    monkeypatch.setenv("S3_BUCKET_DOCS", "")
    config.get_settings.cache_clear()
    assert config.get_settings().storage_backend == "aws"


def test_demo_mode_is_refused_outside_local_storage(monkeypatch) -> None:
    monkeypatch.setenv("STORAGE_BACKEND", "aws")
    monkeypatch.setenv("S3_BUCKET_DOCS", "b")
    monkeypatch.setenv("DEMO_MODE", "true")
    config.get_settings.cache_clear()
    with pytest.raises(ValueError, match="DEMO_MODE"):
        config.get_settings()
