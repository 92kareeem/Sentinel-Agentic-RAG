"""Pydantic AI v2: ``MCPToolset`` over a FastMCP ``StdioTransport``."""

from __future__ import annotations

import asyncio

from _common import ARGUMENTS, check, server_command


async def main() -> None:
    from fastmcp.client.transports import StdioTransport
    from pydantic_ai.mcp import MCPToolset

    toolset = MCPToolset(StdioTransport(command=server_command(), args=[]))
    async with toolset:
        tools = await toolset.list_tools()
        result = await toolset.direct_call_tool("verify_answer", ARGUMENTS)
    assert isinstance(result, dict), result
    check("pydantic-ai", [t.name for t in tools], result["summary"])


asyncio.run(main())
