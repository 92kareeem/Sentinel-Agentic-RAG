# Contributing

Thanks for looking. This repository holds two things with two licenses:

| Part | Where | License | What it is |
|---|---|---|---|
| `sentinel-verify` | [`packages/sentinel-verify/`](packages/sentinel-verify) | Apache-2.0 | The deterministic grounding gate as a dependency-free Python package |
| Sentinel | everything else | AGPL-3.0 | The full platform: API, agent graph, control plane UI, AWS deployment |

Most first contributions land in the package: it is small, pure Python, fully
offline to test, and every rule in it exists because of a real failure that
has a test.

## Set up

```bash
git clone https://github.com/92kareeem/Sentinel-Agentic-RAG.git
cd Sentinel-Agentic-RAG
```

**Package only** (seconds, no Docker, no keys):

```bash
cd packages/sentinel-verify
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q && ruff check . && mypy src
```

**Whole platform:**

```bash
make install          # venv + frontend deps (on Windows, keep the venv outside OneDrive)
make ingest           # index ./corpus
make check            # lint, types, backend and frontend tests: what CI runs
docker compose up --build   # the full stack on http://localhost:5173
```

`docs/LEARNING_PATH.md` walks through the codebase in ten checkpoints.

## Before you open a pull request

- **One change per PR**, from a branch off `main`. Merged with a merge commit.
- **Tests.** A behaviour change comes with a test that fails without it. For
  the grounding gate, write the test the way the existing ones are written:
  a short docstring saying what real answer went wrong and why.
- **Keep the package dependency-free.** CI fails if `sentinel-verify` grows a
  runtime dependency. Adapters for other libraries are duck-typed on purpose.
- **Public Python ≥ 3.10** in the package; the platform targets 3.12.
- `ruff check`, `ruff format --check` (package) and `mypy` must pass.
- **No secrets**, ever: not in code, tests, fixtures or commit history. The
  pre-commit habit here is `git add <paths>`, never `git add -A`.
- If you change a non-obvious design decision, add or amend an ADR in
  `docs/adr/`. Decisions there are written with the measurement behind them.

## Reporting a bug in the gate

The most useful bug report is a failing test: the answer text, the passages,
and what you expected. The issue template asks for exactly that. False
refusals (a correct answer stripped or refused) are as important as misses.

## Security

Please do not open public issues for vulnerabilities. See
[SECURITY.md](SECURITY.md).

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## License of contributions

By contributing you agree that your contributions to `packages/sentinel-verify`
are licensed under Apache-2.0 and contributions elsewhere under AGPL-3.0, the
same terms as the code they modify.
