"""Per-token price table, so every cost number in the system is computed once.

Role in architecture: feeds Completion.cost_usd, which the trace stores and
the telemetry tab sums. Prices are list prices per million tokens at the
time of writing and WILL drift; the LLM_PRICING_OVERRIDES setting lets an
operator correct them without a release:

    LLM_PRICING_OVERRIDES="groq/openai/gpt-oss-20b=0.10:0.50,openai/gpt-4o=2.5:10"

Unknown models cost 0.0 and are reported as such rather than guessed: a
telemetry tile that reads $0.00 for a model you know is not free is a prompt
to add a row, which beats a confident wrong number.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings

# (input $/1M, output $/1M), keyed "provider/model".
_BUILTIN: dict[str, tuple[float, float]] = {
    "groq/openai/gpt-oss-20b": (0.10, 0.50),
    "groq/openai/gpt-oss-120b": (0.15, 0.75),
    "groq/llama-3.3-70b-versatile": (0.59, 0.79),
    "groq/llama-3.1-8b-instant": (0.05, 0.08),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "openai/gpt-4.1": (2.00, 8.00),
    # Local inference has no per-token bill. Electricity is real but not
    # attributable per request, so it is reported as zero on purpose.
}


def _parse_overrides(raw: str) -> dict[str, tuple[float, float]]:
    table: dict[str, tuple[float, float]] = {}
    for pair in filter(None, (p.strip() for p in raw.split(","))):
        key, _, prices = pair.partition("=")
        cin, _, cout = prices.partition(":")
        try:
            table[key.strip()] = (float(cin), float(cout))
        except ValueError:
            continue  # a malformed override is ignored, never a startup failure
    return table


@lru_cache
def _table() -> dict[str, tuple[float, float]]:
    return {**_BUILTIN, **_parse_overrides(get_settings().llm_pricing_overrides)}


def reset_cache() -> None:
    _table.cache_clear()


def price_for(provider: str, model: str) -> tuple[float, float] | None:
    table = _table()
    if provider in ("local",):
        return (0.0, 0.0)
    return table.get(f"{provider}/{model}")


def estimate_cost(provider: str, model: str, tokens_in: int, tokens_out: int) -> float:
    """USD for one call, or 0.0 when the model has no row."""
    prices = price_for(provider, model)
    if prices is None:
        return 0.0
    cin, cout = prices
    return round((tokens_in * cin + tokens_out * cout) / 1_000_000, 8)
