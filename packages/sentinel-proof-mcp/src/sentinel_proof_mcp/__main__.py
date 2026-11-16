"""``sentinel-proof-mcp``: start the server.

sentinel-proof-mcp                       # stdio, what MCP clients launch
sentinel-proof-mcp --transport streamable-http --port 8765
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .server import build_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sentinel-proof-mcp", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="stdio (default) for local clients; streamable-http to serve over HTTP",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default 8765)")
    args = parser.parse_args(argv)

    server = build_server()
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
