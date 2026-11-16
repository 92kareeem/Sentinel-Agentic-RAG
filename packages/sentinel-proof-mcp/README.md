# sentinel-proof-mcp: a hallucination check your agent can call

[![PyPI](https://img.shields.io/pypi/v/sentinel-proof-mcp.svg)](https://pypi.org/project/sentinel-proof-mcp/)
[![Python](https://img.shields.io/pypi/pyversions/sentinel-proof-mcp.svg)](https://pypi.org/project/sentinel-proof-mcp/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/92kareeem/sentinel-rag/blob/main/packages/sentinel-proof-mcp/LICENSE)

An [MCP](https://modelcontextprotocol.io) server that lets Claude Desktop, Claude
Code, Cursor or any MCP client **check a drafted answer against the sources it
was written from** before showing it. Every number must appear in the passage
the sentence cites; a sentence whose citation is wrong is re-pointed to the
passage that supports it; a sentence no passage supports is removed; an answer
that is mostly unsupported is refused with a reason. Then the agent shows
`safe_text`, not the draft.

It runs locally, makes no model or network calls and needs no API key. The
check is [`sentinel-proof`](https://pypi.org/project/sentinel-proof/), the
deterministic grounding gate; this package only puts it behind one MCP tool.
Try the same check in the browser at the
[playground](https://92kareeem.github.io/sentinel-rag/).

<!-- mcp-name: io.github.92kareeem/sentinel-proof -->

## Install

You need [`uv`](https://docs.astral.sh/uv/) (or `pipx`). The clients below
launch the server on demand with `uvx`; nothing else to install.

**Claude Code**

```bash
claude mcp add --scope user sentinel-proof -- uvx sentinel-proof-mcp
```

**Claude Desktop**: `Settings → Developer → Edit Config`, then add to
`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sentinel-proof": {
      "command": "uvx",
      "args": ["sentinel-proof-mcp"]
    }
  }
}
```

**Cursor**: the same block in `.cursor/mcp.json` (project) or
`~/.cursor/mcp.json` (global).

**Any other client**: the command is `uvx sentinel-proof-mcp` over stdio. For
HTTP, `uvx sentinel-proof-mcp --transport streamable-http --port 8765` serves
`http://127.0.0.1:8765/mcp`.

## What the agent gets

One tool, **`verify_answer`**:

| Argument | |
|---|---|
| `answer` | the draft, with citations |
| `passages` | the retrieved texts in the order the model saw them: strings, or `{"text", "id", "source", "section", "page"}` |
| `question` | optional; figures the user typed are not counted as fabrications |
| `citation_style` | `"numeric"` (default): `[1]`, `[2, 3]` are 1-based positions in `passages`; `"tag"`: `[chunk:<id>]` |
| `max_stripped_ratio` | refuse when more than this fraction of claims is stripped (default 0.5) |

It returns structured JSON: `ok`, `safe_text` (what to show), `summary` (one
line telling the model what happened and what to do), `claims` (each sentence
with `supported` / `repaired` / `stripped`, what it cited, what supports it, and
any figures no passage contains), `repaired_citations`, `stripped_ratio`,
`refusal`.

```json
{
  "ok": true,
  "summary": "Verified: 2 of 3 claims supported. 1 citation(s) re-pointed to the right passage. 1 unsupported claim(s) removed. Show the user safe_text.",
  "safe_text": "Customers can get a refund within 30 days [1]. Requests above 500 USD need manager approval [2].",
  "claims": [
    {"text": "Customers can get a refund within 30 days", "status": "supported", "cited": ["p1"], "supported_by": ["p1"], "unsupported_numbers": []},
    {"text": "Requests above 500 USD need manager approval", "status": "repaired", "cited": ["p1"], "supported_by": ["p2"], "unsupported_numbers": []},
    {"text": "Refunds are processed in 2 business days", "status": "stripped", "cited": ["p2"], "supported_by": [], "unsupported_numbers": ["2"]}
  ],
  "repaired_citations": 1,
  "stripped_ratio": 0.333,
  "refusal": null
}
```

One prompt, **`cite_for_verification`**, which tells the model how to draft so
that every sentence can be checked: cite each sentence as `[n]`, copy figures
exactly, reply `INSUFFICIENT_CONTEXT` when the sources do not answer. Use it as
the system instruction of any answer-from-documents step, and the tool will
have something to verify.

One resource, `sentinel-proof://about`, with the versions and the rules.

## A typical turn

1. The agent retrieves passages (its own search tool, a vector store, files).
2. It drafts an answer with `[1]`-style citations (the prompt above).
3. It calls `verify_answer(answer, passages, question)`.
4. It shows `safe_text`. If `ok` is false, `summary` says what was wrong:
   fabricated figures, no citations, or too little support, and what to do
   next (answer only from the passages, retrieve more, or tell the user).

The server's instructions say exactly this, so a client that reads them will
do it without further prompting.

## What it does not do

It checks evidence, not meaning: word overlap and numbers, no entailment
model. A paraphrase that misstates its source and shares its vocabulary passes;
an invented figure or a citation to the wrong passage does not. It is the
deterministic floor under a judge model, not a replacement for one. Details and
the design decision are in the
[sentinel-proof README](https://github.com/92kareeem/sentinel-rag/blob/main/packages/sentinel-proof/README.md).

## Development

```bash
pip install -e ../sentinel-proof -e ".[dev]"
pytest -q            # in-process through the MCP client, plus one real stdio subprocess
ruff check . && mypy src
```

## License

Apache-2.0.
