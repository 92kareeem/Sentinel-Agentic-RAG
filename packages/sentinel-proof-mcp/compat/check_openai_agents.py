"""OpenAI Agents SDK: ``agents.mcp.MCPServerStdio``."""

from __future__ import annotations

import asyncio

from _common import ARGUMENTS, check, server_command


async def main() -> None:
    from agents.mcp import MCPServerStdio

    params = {"command": server_command(), "args": []}
    async with MCPServerStdio(name="sentinel-proof", params=params) as server:
        tools = await server.list_tools()
        result = await server.call_tool("verify_answer", ARGUMENTS)
    assert result.structured_content is not None, result
    check("openai-agents", [t.name for t in tools], result.structured_content["summary"])


asyncio.run(main())
