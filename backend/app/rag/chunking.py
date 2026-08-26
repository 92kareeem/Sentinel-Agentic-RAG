"""Structure-aware chunker.

Role in architecture: converts raw documents (.md/.txt/.pdf) into Chunk records
— the retrieval ground truth stored in chunks.jsonl. Headings become
section_path, tables stay atomic, prose is windowed by token count with
overlap, and chunk ids are deterministic so re-ingestion is idempotent.

The tokenizer is injected as `Offsets = Callable[[str], list[tuple[int, int]]]`
returning (char_start, char_end) per token: production injects the MiniLM fast
tokenizer (embeddings.token_offsets); tests inject a plain word-splitter so the
suite runs offline.
"""

import re
from collections.abc import Callable
from pathlib import Path

from app.models.schemas import Chunk, ChunkType

Offsets = Callable[[str], list[tuple[int, int]]]

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")


def simple_word_offsets(text: str) -> list[tuple[int, int]]:
    """Fallback/test tokenizer: one token per \\w+ run. No model download needed."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


# ---------------------------------------------------------------- structure


def _split_sections(text: str) -> list[tuple[str, str, int]]:
    """Split markdown-ish text on headings.

    Returns (section_path, body, char_offset_of_body) triples. section_path
    joins the active heading stack with " > ". Text before any heading gets
    path "(preamble)".
    """
    lines = text.splitlines(keepends=True)
    stack: list[tuple[int, str]] = []  # (level, title)
    sections: list[tuple[str, str, int]] = []
    buf: list[str] = []
    buf_offset = 0
    offset = 0

    def flush() -> None:
        body = "".join(buf)
        if body.strip():
            path = " > ".join(t for _, t in stack) or "(preamble)"
            sections.append((path, body, buf_offset))
        buf.clear()

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).strip()))
            buf_offset = offset + len(line)
        else:
            if not buf:
                buf_offset = offset
            buf.append(line)
        offset += len(line)
    flush()
    return sections


def _split_blocks(body: str) -> list[tuple[str, bool, int]]:
    """Split a section body into (text, is_table, rel_offset) blocks.

    Consecutive markdown table lines form one atomic table block.
    """
    blocks: list[tuple[str, bool, int]] = []
    lines = body.splitlines(keepends=True)
    cur: list[str] = []
    cur_off = 0
    cur_table = False
    offset = 0

    def flush() -> None:
        text = "".join(cur)
        if text.strip():
            blocks.append((text, cur_table, cur_off))
        cur.clear()

    for line in lines:
        is_table = bool(_TABLE_LINE_RE.match(line))
        if cur and is_table != cur_table:
            flush()
            cur_off = offset
        if not cur:
            cur_off = offset
        cur.append(line)
        cur_table = is_table
        offset += len(line)
    flush()
    return blocks


# ---------------------------------------------------------------- windowing


def _window_prose(
    text: str, offsets: list[tuple[int, int]], size: int, overlap: int
) -> list[tuple[int, int, int]]:
    """Yield (char_start, char_end, token_count) windows of `size` tokens,
    stepping size-overlap tokens, snapped to token boundaries."""
    if not offsets:
        return []
    step = max(size - overlap, 1)
    windows: list[tuple[int, int, int]] = []
    for start_tok in range(0, len(offsets), step):
        end_tok = min(start_tok + size, len(offsets))
        windows.append((offsets[start_tok][0], offsets[end_tok - 1][1], end_tok - start_tok))
        if end_tok == len(offsets):
            break
    return windows


# ---------------------------------------------------------------- public API


def _page_index(text: str) -> list[tuple[int, int]]:
    """Build [(char_offset, page_number)] from the extractor's page markers,
    so any character offset can be mapped back to the page it came from."""
    from app.rag.pdf import PAGE_MARKER_RE

    return [(m.start(), int(m.group(1))) for m in PAGE_MARKER_RE.finditer(text)]


def _strip_page_markers(text: str) -> str:
    """Remove page markers from user-visible chunk text.

    Markers exist only to carry page numbers from the extractor to the chunker;
    leaving them in would put "<<<SENTINEL_PAGE:3>>>" into embeddings, LLM
    context and quoted citations.
    """
    from app.rag.pdf import PAGE_MARKER_RE

    return PAGE_MARKER_RE.sub("", text).strip()


def _page_for_offset(page_marks: list[tuple[int, int]], offset: int) -> int | None:
    """Page number covering a char offset (the last marker at or before it)."""
    if not page_marks:
        return None
    page = page_marks[0][1]
    for mark_offset, mark_page in page_marks:
        if mark_offset > offset:
            break
        page = mark_page
    return page


def chunk_text(
    doc_id: str,
    text: str,
    tokenize: Offsets,
    size: int,
    overlap: int,
    *,
    owner_id: str = "",
    source_filename: str = "",
    max_chunks: int | None = None,
) -> list[Chunk]:
    """Chunk one parsed (markdown-ish) document into Chunk records.

    doc_id is passed in by the caller and is the server-generated document
    identity — it is never derived from the filename here, because filenames
    are neither unique across users nor stable across re-uploads.
    """
    page_marks = _page_index(text)
    chunks: list[Chunk] = []

    for s_idx, (section_path, body, s_off) in enumerate(_split_sections(text)):
        c_idx = 0
        for block_text, is_table, b_off in _split_blocks(body):
            base = s_off + b_off
            page = _page_for_offset(page_marks, base)

            if is_table:  # tables are atomic regardless of size
                spans = [(base, base + len(block_text), len(tokenize(block_text)))]
                texts = [block_text.strip()]
            else:
                offs = tokenize(block_text)
                windows = _window_prose(block_text, offs, size, overlap)
                spans = [(base + s, base + e, n) for s, e, n in windows]
                texts = [block_text[s:e].strip() for s, e, _ in windows]

            for (start, end, tokens), raw in zip(spans, texts, strict=True):
                body_text = _strip_page_markers(raw)
                if not body_text:  # a marker-only block strips to nothing
                    continue
                chunks.append(
                    Chunk(
                        # page is part of the id so re-chunking a document whose
                        # page layout changed can't silently reuse a stale id
                        chunk_id=f"{doc_id}_p{page or 0}_s{s_idx}_c{c_idx}",
                        doc_id=doc_id,
                        owner_id=owner_id,
                        section_path=section_path,
                        text=body_text,
                        is_table=is_table,
                        chunk_type=ChunkType.TABLE if is_table else ChunkType.PROSE,
                        page_number=page,
                        source_filename=source_filename,
                        token_count=tokens,
                        char_start=start,
                        char_end=end,
                    )
                )
                c_idx += 1
                if max_chunks is not None and len(chunks) >= max_chunks:
                    return chunks[:max_chunks]
    return chunks


def pdf_to_markdown(path: Path) -> str:
    """Extract a PDF as page-marked markdown.

    Delegates to rag/pdf.py, which handles reading order, running headers and
    footers, tables, and — importantly — raises PdfExtractionError for
    encrypted/scanned/corrupt input instead of returning "" for the caller to
    index as a successful empty document.
    """
    from app.rag.pdf import blocks_to_markdown, extract_pdf

    return blocks_to_markdown(extract_pdf(path))


def extract_document_text(path: Path) -> str:
    """Parsed text for any supported file type, with page markers for PDFs."""
    if path.suffix.lower() == ".pdf":
        return pdf_to_markdown(path)
    return path.read_text(encoding="utf-8", errors="replace")


def chunk_file(
    path: Path,
    tokenize: Offsets,
    size: int,
    overlap: int,
    *,
    doc_id: str,
    owner_id: str = "",
    source_filename: str = "",
    max_chunks: int | None = None,
) -> list[Chunk]:
    """Chunk a .md/.txt/.pdf file under an explicit, caller-supplied doc_id.

    doc_id is required: deriving it from the filename stem (the previous
    behavior) meant two users uploading "policy.pdf" shared one identity and
    silently overwrote each other's chunks in the index.
    """
    return chunk_text(
        doc_id,
        extract_document_text(path),
        tokenize,
        size,
        overlap,
        owner_id=owner_id,
        source_filename=source_filename or path.name,
        max_chunks=max_chunks,
    )
