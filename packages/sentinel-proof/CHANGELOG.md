# Changelog

All notable changes to `sentinel-proof`. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] - 2026-11-15

Documentation and packaging metadata only; no change to verification behaviour.

### Changed

- README leads with the problems the package solves, and adds a comparison with
  LLM-judge metrics, hallucination classifiers and guardrail frameworks, plus an FAQ.
- PyPI summary, keywords and classifiers describe the package in the terms
  people search for.

### Added

- `tests/test_readme_claims.py`: every behaviour the README promises is tested.

## [0.1.0] - 2026-11-15

First release, extracted from the grounding gate that runs in production
inside [Sentinel](https://github.com/92kareeem/sentinel-rag).

### Added

- `verify()`: per-sentence grounding check with three outcomes (supported,
  repaired, stripped), a strip-ratio refusal threshold, and fail-closed
  behaviour for uncited answers.
- `Verdict` and `Claim` with per-claim detail, including the figures a stripped
  claim contained that no cited passage did.
- Numeric (`[1]`, `[2, 3]`) and tag (`[chunk:id]`) citation styles; lenticular,
  padded and full-width tag shapes are normalised.
- `Passage` with duck-typed adapters `from_langchain` and `from_llamaindex`.
- Provenance hashing (`provenance_hash`, `check_provenance`, `HASH_RECIPE`).
- Refusal reason codes and safe refusal prose that never carries the draft.
- `sentinel-proof check` CLI with human and JSON output.
