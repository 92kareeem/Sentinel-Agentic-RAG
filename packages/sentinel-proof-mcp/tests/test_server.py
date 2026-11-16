"""The server, exercised the way a client would: through the MCP protocol.

In-process (``Client(server)``) for the behaviour, and once over stdio as a
real subprocess, which is how Claude Desktop, Claude Code and Cursor launch
it. The subprocess test is the one that catches a broken entry point.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from mcp import Client, StdioServerParameters
from mcp.types import CallToolResult

from sentinel_proof_mcp import SERVER_NAME, __version__, build_server

PASSAGES = [
    "Customers may request a refund within 30 days of purchase.",
    "Refund requests above 500 USD require manager approval.",
]
ANSWER = (
    "Customers can get a refund within 30 days [1]. "
    "Requests above 500 USD need manager approval [1]. "
    "Refunds are processed in 2 business days [2]."
)


def run(coro: Callable[[Client], Awaitable[Any]]) -> Any:
    async def _main() -> Any:
        async with Client(build_server()) as client:
            return await coro(client)

    return asyncio.run(_main())


def call(client: Client, **arguments: Any) -> Awaitable[CallToolResult]:
    return client.call_tool("verify_answer", arguments)


def test_server_identity_and_instructions() -> None:
    async def go(c: Client) -> tuple[str, str, str | None]:
        assert c.server_info is not None
        return c.server_info.name, c.server_info.version, c.instructions

    name, version, instructions = run(go)
    assert name == SERVER_NAME
    assert version == __version__
    assert instructions and "verify_answer" in instructions and "safe_text" in instructions


def test_tool_is_advertised_with_schema_and_hints() -> None:
    async def go(c: Client) -> Any:
        return (await c.list_tools()).tools

    tools = run(go)
    assert [t.name for t in tools] == ["verify_answer"]
    tool = tools[0]
    assert tool.input_schema["required"] == ["answer", "passages"]
    assert tool.output_schema is not None
    assert {"summary", "safe_text", "claims"} <= set(tool.output_schema["properties"])
    assert tool.annotations is not None
    assert tool.annotations.read_only_hint is True
    assert tool.annotations.open_world_hint is False
    assert tool.description and "stripped" in tool.description


def test_verify_repairs_and_strips() -> None:
    async def go(c: Client) -> CallToolResult:
        return await call(
            c, answer=ANSWER, passages=PASSAGES, question="What is the refund policy?"
        )

    r = run(go)
    assert not r.is_error
    out = r.structured_content
    assert out is not None
    assert out["ok"] is True and out["passed"] is True
    assert [cl["status"] for cl in out["claims"]] == ["supported", "repaired", "stripped"]
    assert out["repaired_citations"] == 1
    assert out["safe_text"] == (
        "Customers can get a refund within 30 days [1]. "
        "Requests above 500 USD need manager approval [2]."
    )
    assert out["summary"].startswith("Verified: 2 of 3 claims supported")
    assert "re-pointed" in out["summary"] and "removed" in out["summary"]
    # The text copy the model reads is the same JSON.
    assert json.loads(r.content[0].text) == out  # type: ignore[union-attr]


def test_refusal_names_the_fabricated_figures_and_hides_the_draft() -> None:
    async def go(c: Client) -> CallToolResult:
        return await call(
            c,
            answer="The premium plan costs $9,999 per year [1]. It has a 7% discount [1].",
            passages=["The premium plan costs $1,200 per year with a 4.5% discount."],
        )

    out = run(go).structured_content
    assert out is not None
    assert out["ok"] is False and out["refusal"] == "UNVERIFIABLE_ANSWER"
    assert out["answer"] == ""
    assert "9999" in out["summary"] and "7" in out["summary"]
    assert "Do not show the draft" in out["summary"]
    assert "9,999" not in out["safe_text"]


def test_no_citations_fails_closed_with_advice() -> None:
    async def go(c: Client) -> CallToolResult:
        return await call(c, answer="Customers can get a refund within 30 days.", passages=PASSAGES)

    out = run(go).structured_content
    assert out is not None
    assert out["ok"] is False and out["claims"] == []
    assert "No citations could be parsed" in out["summary"]
    assert "[1], [2]" in out["summary"]


def test_model_refusal_passes_through() -> None:
    async def go(c: Client) -> CallToolResult:
        return await call(c, answer="INSUFFICIENT_CONTEXT", passages=PASSAGES)

    out = run(go).structured_content
    assert out is not None
    assert out["is_refusal"] is True and out["refusal"] == "INSUFFICIENT_EVIDENCE"
    assert "nothing to verify" in out["summary"]


def test_passage_objects_and_tag_citations() -> None:
    async def go(c: Client) -> CallToolResult:
        return await call(
            c,
            answer="Battery capacity is 5000 mAh [chunk:spec]. It weighs 189 grams [chunk:spec].",
            passages=[
                {
                    "text": "Battery capacity is 5000 mAh.",
                    "id": "spec",
                    "source": "spec.pdf",
                    "page": 3,
                },
                {"text": "The device weighs 189 grams.", "id": "weight"},
            ],
            citation_style="tag",
        )

    out = run(go).structured_content
    assert out is not None
    assert out["ok"] is True
    assert [cl["status"] for cl in out["claims"]] == ["supported", "repaired"]
    assert out["claims"][1]["supported_by"] == ["weight"]
    assert out["safe_text"].endswith("[chunk:weight].")


def test_mixed_strings_and_objects_get_positional_ids() -> None:
    async def go(c: Client) -> CallToolResult:
        return await call(
            c,
            answer="Refunds within 30 days [1]. Approval above 500 USD [2].",
            passages=[PASSAGES[0], {"text": PASSAGES[1]}],
        )

    out = run(go).structured_content
    assert out is not None
    assert out["ok"] is True and out["cited_ids"] == ["p1", "p2"]


def test_question_numbers_are_not_fabrications() -> None:
    async def go(c: Client) -> CallToolResult:
        return await call(
            c,
            answer="No, 45 days is outside the window; refunds are within 30 days [1].",
            passages=PASSAGES,
            question="Can I return it after 45 days?",
        )

    out = run(go).structured_content
    assert out is not None
    assert out["ok"] is True and out["claims"][0]["status"] == "supported"


def test_max_stripped_ratio_is_honoured() -> None:
    async def go(c: Client) -> CallToolResult:
        return await call(c, answer=ANSWER, passages=PASSAGES, max_stripped_ratio=0.2)

    out = run(go).structured_content
    assert out is not None
    assert out["ok"] is False and "limit 20%" in out["summary"]


@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        ({"answer": "x [1].", "passages": []}, "must not be empty"),
        ({"answer": "   ", "passages": PASSAGES}, "answer must not be empty"),
        ({"answer": "x [1].", "passages": [{"id": "a"}]}, "has no text"),
        (
            {"answer": "x [1].", "passages": [{"text": "a", "id": "d"}, {"text": "b", "id": "d"}]},
            "duplicate",
        ),
        ({"answer": "x [1].", "passages": PASSAGES, "max_stripped_ratio": 2}, "between 0 and 1"),
        ({"answer": "x [1].", "passages": "not a list"}, "passages"),
        ({"answer": "x [1].", "passages": PASSAGES, "citation_style": "roman"}, "citation_style"),
    ],
)
def test_bad_input_is_an_error_the_model_can_read(arguments: dict[str, Any], fragment: str) -> None:
    async def go(c: Client) -> CallToolResult:
        return await c.call_tool("verify_answer", arguments)

    r = run(go)
    assert r.is_error
    assert fragment in r.content[0].text  # type: ignore[union-attr]


def test_prompt_and_resource() -> None:
    async def go(c: Client) -> tuple[Any, Any, Any]:
        prompts = (await c.list_prompts()).prompts
        p = await c.get_prompt("cite_for_verification", {"citation_style": "tag"})
        res = await c.read_resource("sentinel-proof://about")
        return prompts, p, res

    prompts, p, res = run(go)
    assert [x.name for x in prompts] == ["cite_for_verification"]
    text = p.messages[0].content.text
    assert "[chunk:<id>]" in text and "INSUFFICIENT_CONTEXT" in text and "verify_answer" in text
    assert res.contents[0].text.startswith(f"sentinel-proof-mcp {__version__} over sentinel-proof ")


def test_stdio_subprocess_end_to_end() -> None:
    """Launch the installed module over stdio, as MCP clients do, and call the tool."""

    async def go() -> tuple[str, CallToolResult]:
        params = StdioServerParameters(command=sys.executable, args=["-m", "sentinel_proof_mcp"])
        async with Client(params) as c:
            assert c.server_info is not None
            r = await c.call_tool("verify_answer", {"answer": ANSWER, "passages": PASSAGES})
            return c.server_info.name, r

    name, r = asyncio.run(go())
    assert name == SERVER_NAME
    assert not r.is_error
    assert r.structured_content is not None and r.structured_content["repaired_citations"] == 1
