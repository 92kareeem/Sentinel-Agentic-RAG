# sentinel-proof-mcp: stop your agent from showing hallucinated answers

[![PyPI](https://img.shields.io/pypi/v/sentinel-proof-mcp.svg)](https://pypi.org/project/sentinel-proof-mcp/)
[![Python](https://img.shields.io/pypi/pyversions/sentinel-proof-mcp.svg)](https://pypi.org/project/sentinel-proof-mcp/)
[![MCP registry](https://img.shields.io/badge/MCP%20registry-io.github.92kareeem%2Fsentinel--proof-6f42c1.svg)](https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.92kareeem/sentinel-proof)
[![No API key](https://img.shields.io/badge/API%20key-none-brightgreen.svg)](#faq)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/92kareeem/sentinel-rag/blob/main/packages/sentinel-proof-mcp/LICENSE)

**An MCP server that fact-checks an LLM answer against the documents it was
written from, before the user sees it.** Claude Desktop, Claude Code, Cursor,
VS Code, Windsurf or any [Model Context Protocol](https://modelcontextprotocol.io)
client gets one tool, `verify_answer`. The agent passes its retrieved passages
and its draft; the tool checks that every number appears in the passage the
sentence cites and that every citation points at a passage that supports it.
Wrong citations are re-pointed, unsupported sentences are removed, and an
answer that is mostly unsupported is refused with a reason. The agent then
shows `safe_text`, not the draft.

Runs locally over stdio. No model call, no network, no API key, no GPU: the
check is [`sentinel-proof`](https://pypi.org/project/sentinel-proof/), a
deterministic grounding gate that takes microseconds, so it can run on every
answer rather than a sample. Nothing you verify leaves your machine.

```bash
claude mcp add --scope user sentinel-proof -- uvx sentinel-proof-mcp
```

Or **[try the same check in your browser](https://92kareeem.github.io/sentinel-rag/)**
first; nothing you paste is uploaded.

<!-- mcp-name: io.github.92kareeem/sentinel-proof -->

**Use it when:**

- your agent **answers from files, a vector store or search results** and
  sometimes makes up a price, date, limit or percentage that is not in them;
- it **cites the wrong source**, so a user who clicks `[2]` reads a passage
  that says nothing of the kind;
- you want the answer **grounded or refused**, not a faithfulness score you
  then have to threshold, and not a second model call that is itself a model;
- you need a check that is **the same every time**, for a regulated domain
  (legal, finance, healthcare, HR policy) or for a test suite;
- you want it **inside the agent loop**, so the model fixes its own answer
  before replying, instead of a dashboard that tells you afterwards.

## Install: works with every MCP client

MCP is an open protocol, and this server is a plain stdio process, so it is
not tied to Claude or to any model. Any client or agent framework that can
launch an MCP server can use it, with any LLM behind it. One prerequisite:
[`uv`](https://docs.astral.sh/uv/) (`pipx` works too). Every client below then
launches the server on demand with `uvx sentinel-proof-mcp`; nothing else to
install.

[![Install in Cursor](https://cursor.com/deeplink/mcp-install-dark.svg)](https://cursor.com/en/install-mcp?name=sentinel-proof&config=eyJjb21tYW5kIjoidXZ4IiwiYXJncyI6WyJzZW50aW5lbC1wcm9vZi1tY3AiXX0=)
[![Install in VS Code](https://img.shields.io/badge/VS_Code-Install_sentinel--proof-0098FF?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=sentinel-proof&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22sentinel-proof-mcp%22%5D%7D)

**One command**, for clients with a CLI:

```bash
claude mcp add --scope user sentinel-proof -- uvx sentinel-proof-mcp   # Claude Code
codex mcp add sentinel-proof -- uvx sentinel-proof-mcp                 # OpenAI Codex CLI
gemini mcp add sentinel-proof uvx sentinel-proof-mcp                   # Gemini CLI
```

**One JSON block**, for clients with a config file. This is the standard
`mcpServers` shape:

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

| Client | Where the block goes |
|---|---|
| Claude Desktop | `claude_desktop_config.json` (`Settings → Developer → Edit Config`) |
| Cursor | `.cursor/mcp.json` in the project, or `~/.cursor/mcp.json` |
| Windsurf | `~/.codeium/windsurf/mcp_config.json` |
| Cline, Roo Code | the MCP settings file the extension opens from its MCP panel |
| LM Studio | `mcp.json` (`Program → Install → Edit mcp.json`) |
| ChatGPT desktop, Goose, Continue, Zed, JetBrains AI, Kiro, OpenCode, Warp | the same `command` and `args`; see each client's MCP settings for the file |
| VS Code (Copilot agent mode) | `.vscode/mcp.json`, with the top-level key `servers` instead of `mcpServers` |

**Over HTTP**, for clients that connect to a URL rather than launching a
process (and for Docker, a shared dev box, or an agent running elsewhere):

```bash
uvx sentinel-proof-mcp --transport streamable-http --host 0.0.0.0 --port 8765
# clients connect to http://<host>:8765/mcp
```

**From Python**, if you would rather manage the environment yourself:
`pip install sentinel-proof-mcp`, then the command is `sentinel-proof-mcp`
(or `python -m sentinel_proof_mcp`).

**Windows note**: if a desktop client cannot find `uvx`, put the full path in
`command` (`where uvx` prints it) or set `uv`'s install directory on `PATH`
for GUI apps.

### From an agent framework

The server works with any LLM provider, because the framework, not the
server, talks to the model. These were run against this release:

**OpenAI Agents SDK**

```python
from agents import Agent, Runner
from agents.mcp import MCPServerStdio

async with MCPServerStdio(name="sentinel-proof", params={"command": "uvx", "args": ["sentinel-proof-mcp"]}) as proof:
    agent = Agent(name="docs-qa", instructions="Answer from the passages, cite [n], then verify_answer before replying.", mcp_servers=[proof])
    result = await Runner.run(agent, "What is the refund policy?")
```

**LangChain / LangGraph**

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "sentinel-proof": {"command": "uvx", "args": ["sentinel-proof-mcp"], "transport": "stdio"},
})
tools = await client.get_tools()   # [verify_answer], ready for create_agent / bind_tools
```

`langchain-mcp-adapters` pins the `mcp` package below 2.0 and this server
needs 2.x. That is why the examples launch it with `uvx`, which gives it its
own environment; the two versions talk over the wire without issue.

**Pydantic AI**

```python
from fastmcp.client.transports import StdioTransport
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

proof = MCPToolset(StdioTransport(command="uvx", args=["sentinel-proof-mcp"]))
agent = Agent("<provider>:<model>", toolsets=[proof])   # any provider Pydantic AI supports
```

Google ADK, LlamaIndex, CrewAI, AutoGen, Semantic Kernel, Mastra, Vercel AI
SDK and n8n all have an MCP client that takes the same `command` and `args`.

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
do it without further prompting. In Claude Desktop or Cursor you can also just
say "verify that against the sources with sentinel-proof".

## How it compares

Other ways to catch a hallucinated answer from inside an MCP agent, and where
this one differs. None of these is wrong; they answer different questions.

| | Where it runs | Needs | What you get back | Fixes the answer |
|---|---|---|---|---|
| **sentinel-proof-mcp** | your machine | nothing | per-sentence verdict, `safe_text`, refusal with reason | yes: re-points citations, drops unsupported sentences |
| Ask the model to check itself | the model | a second model call | an opinion, different each run | sometimes, unverifiably |
| Hosted NLI / "verify" servers | a vendor's API | API key, your text uploaded | a score or label per claim | no |
| URL fact-checkers | the web | network | whether a page says X | no; and your passages are not URLs |
| RAG evaluation frameworks | your CI | a judge model | faithfulness scores over a dataset | no; offline |

The trade: this check is lexical and numeric, not semantic. It catches the
invented figure and the citation to the wrong passage, which are the
hallucinations users notice and screenshot. It does not catch a paraphrase that
misstates its source using the source's own words; for that, put a judge model
behind it. It is the deterministic floor, not the ceiling.

## FAQ

**Does it need an API key or send my documents anywhere?** No. The server is
a local process over stdio, the check is pure Python with no dependencies, and
it makes no network calls. The passages and the answer stay in the process.

**Which clients and models does it work with?** Any MCP client, with any
model. The server never talks to an LLM; the client does. It speaks stdio,
the transport every desktop client and CLI uses to launch local servers, and
streamable HTTP for the rest. The test suite drives it over stdio as a real
subprocess, and the framework examples above were run against this release
with the OpenAI Agents SDK, LangChain (on the older `mcp` 1.x) and Pydantic
AI.

**How is this different from asking Claude to double-check its answer?** The
model checking itself is another model call: it costs tokens, takes seconds,
and gives a different answer each time. This tool is a rule: a number is in
the cited passage or it is not. It returns the same verdict every run, in
microseconds, and tells the model exactly which sentence failed and why.

**My agent does not cite. Will it still work?** Not yet: with nothing cited
there is nothing to check, and the tool says so (`ok: false`, "No citations
could be parsed"). Attach the `cite_for_verification` prompt to the drafting
step, or add "cite each sentence as [n]" to your system prompt. Citation is
what makes the answer checkable.

**What counts as a passage?** Whatever the model saw: chunks from a vector
store, paragraphs from a file, search snippets, tool results. Pass them in the
order the model saw them so that `[1]` means the first. Objects with `id`,
`source`, `section` and `page` let the verdict say which document a sentence
came from.

**Does it work in languages other than English?** Numbers are checked in any
language. The word-overlap rule uses an English stop-word list, so for other
languages it is stricter than intended; multilingual stop-words are on the
roadmap of `sentinel-proof`.

**Is there a version I can call from Python without MCP?**
[`sentinel-proof`](https://pypi.org/project/sentinel-proof/) is the library;
`verify(answer, passages)` is the same check. This package only puts it
behind an MCP tool.

**Where is it listed?** In the official MCP registry as
[`io.github.92kareeem/sentinel-proof`](https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.92kareeem/sentinel-proof),
and on [PyPI](https://pypi.org/project/sentinel-proof-mcp/).

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
