#!/bin/sh
# Index the shipped corpus on first start, then serve.
#
# Idempotent: the published index lives in the data volume, so a restart
# reuses it and an upload made through the UI is never wiped by a rebuild.
# Delete the volume (`docker compose down -v`) to start from the corpus again.
set -eu

if [ ! -f "${INDEX_DIR}/current.json" ]; then
  echo "sentinel: no index in ${INDEX_DIR}; indexing /app/corpus (first start only)"
  python -m ingestion.ingest /app/corpus
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers --no-access-log
