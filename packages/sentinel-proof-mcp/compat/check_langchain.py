"""LangChain / LangGraph: ``langchain_mcp_adapters.MultiServerMCPClient``.

The adapter pins ``mcp`` below 2.0 while the server needs 2.x, so this check
runs in its own environment and talks to the installed server over stdio,
which is how users run it too.
"""

from __future__ import annotations

import asyncio
import json

from _common import ARGUMENTS, check, server_command


async def main() -> None:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {"sentinel-proof": {"command": server_command(), "args": [], "transport": "stdio"}}
    )
    tools = await client.get_tools()
    result = await tools[0].ainvoke(ARGUMENTS)
    # The adapter returns the tool's text content: a list of content blocks or a string.
    text = result[0]["text"] if isinstance(result, list) else str(result)
    check("langchain", [t.name for t in tools], json.loads(text)["summary"])


asyncio.run(main())
