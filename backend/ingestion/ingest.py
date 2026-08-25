"""Ingestion CLI: build the hybrid index from a folder of documents.

Usage:
    python -m ingestion.ingest ./docs            # chunk + embed + write index/
    python -m ingestion.ingest --smoke-only "your question"   # query existing index

Role in architecture: the offline half of RAG. Runs on your laptop (or CI),
never in Lambda. Full rebuild each run + deterministic chunk ids = idempotent.
"""

import argparse
import re
import sys
import time
from pathlib import Path

from app.config import get_settings
from app.models.schemas import Chunk
from app.rag import bm25_store, embeddings, index_store
from app.rag.pdf import PdfExtractionError

SUPPORTED = {".md", ".txt", ".pdf"}


def _corpus_doc_id(path: Path) -> str:
    """Stable identity for a file in the seeded corpus.

    Uploaded documents get a server-generated uuid; the seeded demo corpus is
    an operator-controlled folder where a slug of the filename is both stable
    across rebuilds and readable in traces. These never collide with uploads
    (uuid4 hex) and are owned by no user, so they act as a shared corpus.
    """
    return re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")


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
        try:
            doc_chunks = chunk_file(
                path,
                embeddings.token_offsets,
                s.chunk_size_tokens,
                s.chunk_overlap_tokens,
                doc_id=_corpus_doc_id(path),
                owner_id="",  # shared corpus: visible to all callers
                source_filename=path.name,
                max_chunks=s.max_chunks_per_document,
            )
        except PdfExtractionError as exc:
            # One unreadable file must not abort a corpus rebuild.
            print(f"  {path.name}: SKIPPED ({exc.code.value}: {exc.message})")
            continue
        chunks.extend(doc_chunks)
        n_tables = sum(c.is_table for c in doc_chunks)
        pages = {c.page_number for c in doc_chunks if c.page_number}
        page_note = f", {len(pages)} pages" if pages else ""
        print(f"  {path.name}: {len(doc_chunks)} chunks ({n_tables} tables{page_note})")

    if not chunks:
        sys.exit("no chunks produced — nothing to index")

    texts = [c.embed_text for c in chunks]
    print(f"[embed] {len(chunks)} chunks with {s.embed_model_name} ...")
    vecs = embeddings.embed_texts(texts)

    index = index_store.build_faiss(vecs)
    bm25 = bm25_store.build(texts)
    # A CLI rebuild replaces the corpus wholesale, so it publishes on top of
    # whatever is current rather than merging into it.
    version = index_store.publish(
        index, chunks, bm25, base_version=index_store.read_pointer(Path(s.index_dir))
    )

    counts = sorted(c.token_count for c in chunks)
    pct = lambda p: counts[min(int(len(counts) * p), len(counts) - 1)]  # noqa: E731
    print(
        f"[stats] token_count p50={pct(0.5)} p90={pct(0.9)} max={counts[-1]} | "
        f"{time.perf_counter() - t0:.1f}s total"
    )
    print(f"[done] published {version} in {s.index_dir}/")


def smoke(query: str) -> None:
    """Hybrid-search the built index and print top-3 — the P1 definition of done."""
    s = get_settings()
    loaded = index_store.load_current(Path(s.index_dir))
    if loaded is None:
        sys.exit("no index published yet — run the ingest command first")
    index, chunks, version = loaded
    bm25 = index_store.load_bm25(version, Path(s.index_dir))

    qvec = embeddings.embed_texts([query])[0]
    dense = [row for row, _ in index_store.search(index, qvec, s.candidates_per_retriever)]
    sparse = [row for row, _ in bm25_store.search(bm25, query, s.candidates_per_retriever)]
    fused = bm25_store.rrf_fuse([dense, sparse], k=s.rrf_k, top_k=3)

    print(f'[smoke] "{query}" -> top-3 of {len(chunks)} chunks in {version}:')
    for rank, (row, score) in enumerate(fused, 1):
        c = chunks[row]
        preview = " ".join(c.text.split())[:160]
        page = f" p{c.page_number}" if c.page_number else ""
        print(f"  {rank}. [{c.chunk_id}] rrf={score:.4f} | {c.section_path}{page}\n     {preview}")


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
