<!-- Thanks. Keep it to one change; open another PR for the next one. -->

## What this changes

<!-- One paragraph: the behaviour before, the behaviour after, and why. -->

## How it was verified

<!-- The commands you ran and what they reported, e.g.
     packages/sentinel-proof: pytest -q (57 passed), ruff, mypy
     make check
     docker compose up --build, then the chaos demo -->

## Checklist

- [ ] A test fails without this change and passes with it
- [ ] `ruff check`, `ruff format --check` (package) and `mypy` pass
- [ ] If this touches `packages/sentinel-proof`: no new runtime dependency, `CHANGELOG.md` updated under *Unreleased*
- [ ] If this changes a design decision: ADR added or amended in `docs/adr/`
- [ ] No secrets, keys or personal data in the diff
