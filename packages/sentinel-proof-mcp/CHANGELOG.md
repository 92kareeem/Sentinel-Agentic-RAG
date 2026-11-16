# Changelog

All notable changes to `sentinel-proof-mcp`. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] - 2026-11-16

Documentation and metadata only; the server is unchanged.

### Changed

- README leads with the problem, adds "use it when", install blocks for VS
  Code and Windsurf, a comparison with other ways to check an answer from an
  MCP agent, and a FAQ.
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
