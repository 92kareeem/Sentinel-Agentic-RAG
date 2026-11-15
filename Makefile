# Sentinel task runner.
#
#   make up        the whole stack in docker (no Python, Node or AWS needed)
#   make dev       backend + frontend on your machine, hot reload
#   make check     everything CI runs: lint, types, backend + frontend tests
#
# On Windows run from Git Bash. The venv lives OUTSIDE OneDrive (sync corrupts
# native DLLs such as torch's c10.dll). Override with: make VENV=/path <target>

ifeq ($(OS),Windows_NT)
VENV ?= C:/venvs/sentinel
PY := $(VENV)/Scripts/python.exe
else
VENV ?= .venv
PY := $(VENV)/bin/python
endif

.PHONY: help up down reset logs demo dev serve web venv install ingest smoke test \
        test-web lint check eval regressions deploy bench bench-up bench-down

help:
	@grep -E '^# ' Makefile | head -6 | sed 's/^# //'

# ------------------------------------------------------------------ docker

up:
	docker compose up --build -d
	@echo "Sentinel: http://localhost:5173   API: http://localhost:8000/docs"

down:
	docker compose down

reset:  ## drop the data volume: next `make up` re-indexes ./corpus from scratch
	docker compose down -v

logs:
	docker compose logs -f api

demo: up
	@echo "Open http://localhost:5173 and click 'Chaos demo' in the header."

# ------------------------------------------------------------------ local

venv:
	python -m venv $(VENV)

install: venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ./packages/sentinel-verify -e ".[dev]"
	cd frontend && npm ci

# Build FAISS + BM25 + chunks.jsonl from ./corpus into ./index.
# ./corpus, not ./docs: the ingester takes every file under the folder, and
# ./docs is this repository's own engineering documentation.
ingest:
	$(PY) -m ingestion.ingest ./corpus

smoke:
	$(PY) -m ingestion.ingest --smoke-only "What are the refund conditions?"

serve:
	DEMO_MODE=true $(PY) -m uvicorn app.main:app --reload --port 8000 --app-dir backend

web:
	cd frontend && VITE_API_BASE=http://localhost:8000 VITE_API_KEY=demo-local npm run dev

dev:
	@echo "Run in two terminals:  make serve   and   make web"

# ------------------------------------------------------------------ quality

test:
	$(PY) -m pytest backend/tests -q

test-web:
	cd frontend && npx tsc --noEmit && npm test

lint:
	$(PY) -m ruff check backend
	$(PY) -m mypy backend/app

check: lint test test-web

eval:
	$(PY) evals/run.py

regressions:
	$(PY) evals/run.py --regressions-only

deploy:
	bash infra/deploy.sh

# ------------------------------------------------------------------ load test
# JMeter 5.6+ and Java 17+ on this machine; see benchmarks/README.md.
# BUILDX_BAKE_ENTITLEMENTS_FS=0: Compose v5 on Windows otherwise refuses to
# read a Dockerfile in a subdirectory of the build context.

BENCH := docker compose -f benchmarks/compose.bench.yml

bench-up:
	BUILDX_BAKE_ENTITLEMENTS_FS=0 $(BENCH) up -d --build --wait

bench-down:
	$(BENCH) down -v

bench: bench-up
	$(PY) benchmarks/run.py $(ARGS)
