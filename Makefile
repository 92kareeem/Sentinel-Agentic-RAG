# Sentinel task runner. Run from Git Bash on Windows: `make install`, `make ingest`, ...
# Venv lives OUTSIDE OneDrive on Windows (OneDrive sync corrupts/locks native DLLs like
# torch's c10.dll). Override with: make VENV=/path/to/venv <target>

ifeq ($(OS),Windows_NT)
VENV ?= C:/venvs/sentinel
PY := $(VENV)/Scripts/python.exe
else
VENV ?= .venv
PY := $(VENV)/bin/python
endif

.PHONY: venv install ingest smoke serve test lint eval deploy

venv:
	python -m venv $(VENV)

install: venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

# Build FAISS + BM25 + chunks.jsonl from ./docs into ./index
ingest:
	$(PY) -m ingestion.ingest ./docs

# Retrieval-only smoke test against the built index
smoke:
	$(PY) -m ingestion.ingest --smoke-only "What are the refund conditions?"

serve:
	$(PY) -m uvicorn app.main:app --reload --port 8000 --app-dir backend

test:
	$(PY) -m pytest backend/tests -q

lint:
	$(PY) -m ruff check backend
	$(PY) -m mypy backend/app

eval:
	$(PY) evals/run.py

deploy:
	bash infra/deploy.sh
