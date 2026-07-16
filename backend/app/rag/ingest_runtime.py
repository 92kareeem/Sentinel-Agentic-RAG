"""In-request document ingestion (the online half of RAG, runs in Lambda).

Role in architecture: the offline `ingestion/ingest.py` rebuilds the whole
index on a laptop; this merges ONE uploaded document into the live index
without a redeploy. Old vectors are reused via FAISS reconstruct (only the new
document is embedded), chunks are appended, BM25 is rebuilt from all texts, and
the artifacts are pushed back to S3. Re-uploading the same filename replaces
that document's chunks (deterministic chunk ids make it idempotent).

Single-writer by design: concurrent uploads could clobber each other's merge.
Acceptable for a demo; a production version would take a lock or serialize
through a queue (forbidden here on free-tier grounds).
"""

from pathlib import Path

import numpy as np

from app.config import get_settings
from app.rag import bm25_store, embeddings, faiss_store
from app.rag.chunking import chunk_file


def _index_dir() -> Path:
    s = get_settings()
    if s.use_s3_index:
        return faiss_store.load_from_s3(Path("/tmp/index"))
    return s.index_dir


def merge_document(local_path: Path) -> tuple[int, str]:
    """Chunk + embed one file and merge it into the index. Returns
    (chunks_indexed, index_version)."""
    s = get_settings()
    index_dir = _index_dir()
    index, chunks = faiss_store.load(index_dir)

    new_chunks = chunk_file(
        local_path, embeddings.token_offsets, s.chunk_size_tokens, s.chunk_overlap_tokens
    )
    if not new_chunks:
        raise ValueError("document produced no chunks (empty or unreadable)")
    doc_id = new_chunks[0].doc_id

    # reuse existing vectors (row i of index == line i of chunks.jsonl), dropping
    # any prior version of this doc so re-upload is a replace, not a duplicate
    old_vecs = index.reconstruct_n(0, index.ntotal) if index.ntotal else np.zeros((0, 384))
    keep = [i for i, c in enumerate(chunks) if c.doc_id != doc_id]
    kept_chunks = [chunks[i] for i in keep]
    kept_vecs = old_vecs[keep] if keep else np.zeros((0, old_vecs.shape[1] or 384))

    new_vecs = embeddings.embed_texts([c.embed_text for c in new_chunks])
    all_chunks = kept_chunks + new_chunks
    all_vecs = np.vstack([kept_vecs, new_vecs]).astype(np.float32)

    faiss_store.save(faiss_store.build_index(all_vecs), all_chunks, index_dir)
    bm25_store.save(bm25_store.build([c.embed_text for c in all_chunks]), index_dir)
    if s.use_s3_index:
        faiss_store.sync_to_s3(index_dir)

    version = str(int((index_dir / faiss_store.INDEX_FILE).stat().st_mtime))
    return len(new_chunks), version
