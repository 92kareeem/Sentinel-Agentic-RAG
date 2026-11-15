"""OpenAI, Azure OpenAI, and OpenAI-compatible local servers (Ollama, vLLM).

Role in architecture: one adapter for every endpoint that speaks the
chat-completions wire format. The three differ only in URL and auth header:

  openai  POST {base_url}/chat/completions          Authorization: Bearer
  azure   POST {endpoint}/openai/deployments/{dep}/chat/completions?api-version=…
          api-key: …   (the "model" is a DEPLOYMENT name, chosen in the portal)
  local   POST {local_llm_base_url}/chat/completions, any bearer token

Keeping them in one module means a fix to the response parser or the
json-mode flag lands on all three at once.
"""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.llm.http_chat import HttpRequest, parse_openai_style, post_chat


def _body(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


def chat_completion(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    json_mode: bool = False,
    deadline_ts: float | None = None,
    max_retries: int | None = None,
    flavour: str = "openai",
) -> tuple[str, int, int]:
    settings = get_settings()
    body = _body(model, messages, temperature, max_tokens, json_mode)

    if flavour == "azure":
        if not settings.azure_openai_endpoint:
            raise ValueError("LLM_PROVIDER=azure requires AZURE_OPENAI_ENDPOINT")
        url = (
            f"{settings.azure_openai_endpoint.rstrip('/')}/openai/deployments/{model}"
            f"/chat/completions?api-version={settings.azure_openai_api_version}"
        )
        headers = {"api-key": settings.openai_api_key, "Content-Type": "application/json"}
        # Azure identifies the model by the URL path; the body field is ignored
        # but harmless. Removed so a deployment named unlike its model does
        # not confuse anyone reading a captured request.
        body.pop("model", None)
        label = "Azure OpenAI"
    elif flavour == "local":
        url = f"{settings.local_llm_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.local_llm_api_key}",
            "Content-Type": "application/json",
        }
        label = "local LLM"
    else:
        url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        label = "OpenAI"

    return post_chat(
        breaker_key=f"{flavour}:{model}",
        request=HttpRequest(url=url, headers=headers, body=body),
        parse=parse_openai_style,
        deadline_ts=deadline_ts,
        max_retries=max_retries,
        provider_label=label,
    )
