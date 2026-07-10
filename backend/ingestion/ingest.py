"""Ingestion CLI: build the hybrid index from a folder of documents.

Usage:
    python -m ingestion.ingest ./docs            # chunk + embed + write index/
    python -m ingestion.ingest --smoke-only "your question"   # query existing index

Role in architecture: the offline half of RAG. Runs on your laptop (or CI),
never in Lambda. Full rebuild each run + deterministic chunk ids = idempotent.
"""

import argparse
import sys
import time
from pathlib import Path

from app.config import get_settings
from app.models.schemas import Chunk
from app.rag import bm25_store, embeddings, faiss_store

SUPPORTED = {".md", ".txt", ".pdf"}


def ingest_folder(folder: Path) -> None:
    from app.rag.chunking import chunk_file  # after arg parsing so --help stays instant

    s = get_settings()
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        sys.exit(f"no ingestable files ({', '.join(SUPPORTED)}) under {folder}")

    print(f"[ingest] {len(files)} file(s) from {folder}")
    t0 = time.perf_counter()
    chunks: list[Chunk] = []
    for path in files:
        doc_chunks = chunk_file(
            path, embeddings.token_offsets, s.chunk_size_tokens, s.chunk_overlap_tokens
        )
        chunks.extend(doc_chunks)
        n_tables = sum(c.is_table for c in doc_chunks)
        print(f"  {path.name}: {len(doc_chunks)} chunks ({n_tables} tables)")

    texts = [c.embed_text for c in chunks]
    print(f"[embed] {len(chunks)} chunks with {s.embed_model_name} ...")
    vecs = embeddings.embed_texts(texts)

    index = faiss_store.build_index(vecs)
    faiss_store.save(index, chunks, s.index_dir)
    bm25_store.save(bm25_store.build(texts), s.index_dir)

    counts = sorted(c.token_count for c in chunks)
    pct = lambda p: counts[min(int(len(counts) * p), len(counts) - 1)]  # noqa: E731
    print(
        f"[stats] token_count p50={pct(0.5)} p90={pct(0.9)} max={counts[-1]} | "
        f"{time.perf_counter() - t0:.1f}s total"
    )
    print(f"[done] artifacts in {s.index_dir}/ (faiss.index, bm25.pkl, chunks.jsonl)")


def smoke(query: str) -> None:
    """Hybrid-search the built index and print top-3 — the P1 definition of done."""
    s = get_settings()
    index, chunks = faiss_store.load(s.index_dir)
    bm25 = bm25_store.load(s.index_dir)

    qvec = embeddings.embed_texts([query])[0]
    dense = [row for row, _ in faiss_store.search(index, qvec, s.candidates_per_retriever)]
    sparse = [row for row, _ in bm25_store.search(bm25, query, s.candidates_per_retriever)]
    fused = bm25_store.rrf_fuse([dense, sparse], k=s.rrf_k, top_k=3)

    print(f'[smoke] "{query}" -> top-3 of {len(chunks)} chunks (RRF over dense+sparse):')
    for rank, (row, score) in enumerate(fused, 1):
        c = chunks[row]
        preview = " ".join(c.text.split())[:160]
        print(f"  {rank}. [{c.chunk_id}] rrf={score:.4f} | {c.section_path}\n     {preview}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build or smoke-test the Sentinel index")
    ap.add_argument("folder", nargs="?", default="./docs", help="docs folder to ingest")
    ap.add_argument("--smoke-only", metavar="QUERY", help="skip build; query existing index")
    args = ap.parse_args()

    if args.smoke_only:
        smoke(args.smoke_only)
    else:
        ingest_folder(Path(args.folder))
        smoke("What are the refund conditions?")  # built-in retrieval smoke test


if __name__ == "__main__":
    main()
