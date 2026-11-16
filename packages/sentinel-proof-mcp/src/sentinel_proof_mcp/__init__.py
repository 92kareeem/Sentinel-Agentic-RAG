"""sentinel-proof as an MCP server.

One tool, ``verify_answer``, checks a drafted answer against the passages it
cites with the deterministic grounding gate from :mod:`sentinel_proof`, and
returns the verdict plus the text that is safe to show. Local, no API key,
no model.
"""

__version__ = "0.1.0"

from .server import SERVER_NAME, build_server  # noqa: E402  (needs __version__)

__all__ = ["SERVER_NAME", "__version__", "build_server"]
