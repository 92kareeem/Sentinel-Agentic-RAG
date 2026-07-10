# Sentinel task runner. Run from Git Bash on Windows: `make install`, `make ingest`, ...
# Venv lives INSIDE the project (.venv). Windows venvs put python under Scripts/.

VENV := .venv
ifeq ($(OS),Windows_NT)
PY := $(VENV)/Scripts/python.exe
else
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
