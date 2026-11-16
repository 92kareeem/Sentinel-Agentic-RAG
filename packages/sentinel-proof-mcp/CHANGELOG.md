# Changelog

All notable changes to `sentinel-proof-mcp`. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.2] - 2026-11-16

Documentation and CI only; the server is unchanged.

### Added

- `compat/`: one script per agent framework (OpenAI Agents SDK, Pydantic AI,
  LangChain/LangGraph) that launches the installed server through that
  framework's MCP client and verifies an answer. A nightly workflow runs them
  against the latest release of each framework and of this package, plus the
  official MCP Inspector as a reference client.

### Changed

- README install section covers every client: one-click links for Cursor and
  VS Code, `claude`/`codex`/`gemini` CLI commands, the standard `mcpServers`
  block with a table of where it goes per client, HTTP for remote clients,
  and the framework snippets above. (Written for 0.1.1 but merged after its
  tag, so it ships here.)

## [0.1.1] - 2026-11-16

Documentation and metadata only; the server is unchanged.

### Changed

- README leads with the problem, adds "use it when", a comparison with other
  ways to check an answer from an MCP agent, and a FAQ.
- PyPI description and keywords name the clients and the problem.
- The README's JSON example is pinned by a test to the tool's real output.

## [0.1.0] - 2026-11-16

First release.

### Added

- `verify_answer` tool: checks a drafted answer against its passages with the
  `sentinel-proof` grounding gate and returns the verdict as structured JSON
  with per-claim detail, a one-line `summary` and the `safe_text` to show.
  Passages as strings or objects; numeric or tag citations; question-aware;
  configurable strip limit. Read-only, idempotent, no network.
- `cite_for_verification` prompt: how to draft so every sentence is checkable.
- `sentinel-proof://about` resource.
- `sentinel-proof-mcp` command: stdio by default, `--transport streamable-http`
  for HTTP.
