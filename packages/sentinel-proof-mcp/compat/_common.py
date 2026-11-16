"""Shared pieces of the framework compatibility checks.

Each ``compat/<framework>.py`` launches the *installed* ``sentinel-proof-mcp``
command through that framework's own MCP client, lists the tools and calls
``verify_answer`` once. They run nightly in CI against the latest release of
each framework and the latest release on PyPI, so a framework renaming its MCP
client (as Pydantic AI did in v2) shows up as a red build, not a bug report.

Run one by hand: ``python compat/check_openai_agents.py [path-to-sentinel-proof-mcp]``.
"""

from __future__ import annotations

import sys

ANSWER = "Refunds within 30 days [1]. Approval above 500 USD [1]."
PASSAGES = ["Refunds are allowed within 30 days.", "Amounts above 500 USD need approval."]
ARGUMENTS = {"answer": ANSWER, "passages": PASSAGES}


def server_command() -> str:
    return sys.argv[1] if len(sys.argv) > 1 else "sentinel-proof-mcp"


def check(framework: str, tool_names: list[str], summary: str) -> None:
    assert tool_names == ["verify_answer"], tool_names
    assert summary.startswith("Verified: 2 of 2 claims supported"), summary
    assert "1 citation(s) re-pointed" in summary, summary
    print(f"{framework}: ok. {summary}")
