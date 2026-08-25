"""In-request document ingestion: parse -> chunk -> embed -> publish.

Role in architecture: turns one uploaded document into queryable chunks and
records the outcome in the document registry. The registry is updated at every
lifecycle transition, so the API can always answer "what happened to my
document?" without inspecting the index.

Concurrency: index publication is serialized by a process-level lock plus a
compare-and-set on the version pointer (see rag/index_store.py). A writer whose
base version was superseded rebuilds against the new head instead of
overwriting it — the previous read-modify-write could silently drop a
concurrently-uploaded document.

Note on the previous background-thread design: `threading.Thread(daemon=True)`
is not a durable job system. On Lambda the runtime is frozen the moment the
handler returns, so a "queued" ingestion could simply never run and the
document would sit in PROCESSING forever. Ingestion here is synchronous and
transactional against the registry; the durable-queue design for AWS is
described in infra/aws_setup.md.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.documents import registry
from app.models.schemas import Chunk, DocumentErrorCode, DocumentStatus
from app.rag import bm25_store, embeddings, index_store
from app.rag.chunking import chunk_file
from app.rag.pdf import PdfExtractionError

# Serializes publication within this process. Cross-process/-instance safety
# comes from the pointer compare-and-set in index_store.publish().
_publish_lock = threading.RLock()

_MAX_PUBLISH_ATTEMPTS = 3

# all-MiniLM-L6-v2's output width; used only when there is no existing index to
# read the true dimension from.
_EMBED_DIM = 384


def _index_root() -> Path:
    settings = get_settings()
    if settings.use_s3_index:
        dest = Path("/tmp/index")
        index_store.load_current_from_s3(dest)
        return dest
    return Path(settings.index_dir)


def _load_current_chunks(root: Path) -> tuple[list[Chunk], np.ndarray, str | None]:
    """Existing chunks + their vectors, so unchanged documents are not re-embedded."""
    loaded = index_store.load_current(root)
    if loaded is None:
        return [], np.zeros((0, _EMBED_DIM), dtype=np.float32), None
    index, chunks, version = loaded
    # Take the width from the index itself rather than assuming the current
    # model's dimensionality: an empty index still knows its own `d`, and
    # hardcoding a size here silently breaks any other embedding model.
    dim = getattr(index, "d", _EMBED_DIM)
    vectors = (
        index.reconstruct_n(0, index.ntotal)
        if index.ntotal
        else np.zeros((0, dim), dtype=np.float32)
    )
    return chunks, vectors, version


def _rebuild_and_publish(root: Path, doc_id: str, new_chunks: list[Chunk]) -> tuple[int, str]:
    """Merge one document's chunks into the current index and publish a new version.

    Retries on ConcurrentUpdateError: losing the pointer race means another
    document was published first, so this rebuild is redone on top of it rather
    than discarding it.
    """
    for attempt in range(_MAX_PUBLISH_ATTEMPTS):
        try:
            # The lock spans read -> rebuild -> publish, not just publish: this
            # whole sequence is the read-modify-write critical section, and
            # holding the lock only over the final write would let two threads
            # both rebuild from the same base and race, which is exactly the
            # lost-update bug being fixed. Embedding inside the lock serializes
            # concurrent uploads within a process — correct for a single-writer
            # index, and far cheaper than losing a user's document.
            with _publish_lock:
                existing_chunks, existing_vecs, base_version = _load_current_chunks(root)

                # Drop any prior generation of THIS document so re-upload
                # replaces rather than duplicates. Matched on document_id,
                # never on filename.
                keep = [i for i, c in enumerate(existing_chunks) if c.doc_id != doc_id]
                kept_chunks = [existing_chunks[i] for i in keep]
                dim = existing_vecs.shape[1] if existing_vecs.shape[1] else _EMBED_DIM
                kept_vecs = (
                    existing_vecs[keep] if keep else np.zeros((0, dim), dtype=np.float32)
                )

                if new_chunks:
                    new_vecs = np.asarray(
                        embeddings.embed_texts([c.embed_text for c in new_chunks]),
                        dtype=np.float32,
                    )
                else:
                    new_vecs = np.zeros((0, dim), dtype=np.float32)

                all_chunks = kept_chunks + new_chunks
                all_vecs = np.vstack([kept_vecs, new_vecs]).astype(np.float32)

                version = index_store.publish(
                    index_store.build_faiss(all_vecs),
                    all_chunks,
                    bm25_store.build([c.embed_text for c in all_chunks]),
                    base_version=base_version,
                    root=root,
                )

            if get_settings().use_s3_index:
                index_store.sync_version_to_s3(version, root)
            index_store.prune_old_versions(root=root)
            return len(new_chunks), version
        except index_store.ConcurrentUpdateError:
            # Another PROCESS (or Lambda instance) published first; rebuild on
            # the new head rather than clobbering it.
            if attempt == _MAX_PUBLISH_ATTEMPTS - 1:
                raise
            continue

    raise RuntimeError("unreachable")  # pragma: no cover


def ingest_document(
    local_path: Path,
    *,
    document_id: str,
    owner_id: str,
    source_filename: str,
) -> tuple[int, str]:
    """Full ingestion for one registered document.

    Every exit path writes a terminal registry state, so a document is never
    left claiming PROCESSING after the request ends.
    """
    settings = get_settings()
    registry.update_status(document_id, owner_id, DocumentStatus.PROCESSING)

    try:
        try:
            chunks = chunk_file(
                local_path,
                embeddings.token_offsets,
                settings.chunk_size_tokens,
                settings.chunk_overlap_tokens,
                doc_id=document_id,
                owner_id=owner_id,
                source_filename=source_filename,
                max_chunks=settings.max_chunks_per_document,
            )
        except PdfExtractionError as exc:
            registry.update_status(
                document_id,
                owner_id,
                DocumentStatus.FAILED,
                error_code=exc.code,
                error_message=exc.message,
            )
            raise

        if not chunks:
            message = "document produced no chunks (empty or unreadable)"
            registry.update_status(
                document_id,
                owner_id,
                DocumentStatus.FAILED,
                error_code=DocumentErrorCode.NO_EXTRACTABLE_TEXT,
                error_message=message,
            )
            raise PdfExtractionError(DocumentErrorCode.NO_EXTRACTABLE_TEXT, message)

        root = _index_root()
        chunk_count, version = _rebuild_and_publish(root, document_id, chunks)

        pages = [c.page_number for c in chunks if c.page_number is not None]
        registry.update_status(
            document_id,
            owner_id,
            DocumentStatus.INDEXED,
            chunk_count=chunk_count,
            page_count=max(pages) if pages else None,
            index_version=version,
            parser_version=settings.parser_version,
            embedding_model=settings.embed_model_name,
        )
        return chunk_count, version

    except PdfExtractionError:
        raise
    except Exception as exc:
        registry.update_status(
            document_id,
            owner_id,
            DocumentStatus.FAILED,
            error_code=DocumentErrorCode.INTERNAL_ERROR,
            error_message=str(exc)[:500],
        )
        raise


def delete_document(document_id: str, owner_id: str) -> str | None:
    """Purge a document's chunks from the index and tombstone its record.

    Returns the new index version, or None if the document had no chunks
    published (e.g. it failed before indexing).
    """
    record = registry.get(document_id, owner_id=owner_id)
    if record is None:
        return None

    version: str | None = None
    if record.status == DocumentStatus.INDEXED:
        _, version = _rebuild_and_publish(_index_root(), document_id, [])
    registry.mark_deleted(document_id, owner_id)
    return version
