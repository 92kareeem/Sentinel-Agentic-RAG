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

from app.models.schemas import Chunk

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


def chunk_text(doc_id: str, text: str, tokenize: Offsets, size: int, overlap: int) -> list[Chunk]:
    """Chunk one parsed (markdown-ish) document into Chunk records."""
    chunks: list[Chunk] = []
    for s_idx, (path, body, s_off) in enumerate(_split_sections(text)):
        c_idx = 0
        for block_text, is_table, b_off in _split_blocks(body):
            base = s_off + b_off
            if is_table:  # tables are atomic regardless of size
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}_s{s_idx}_c{c_idx}",
                        doc_id=doc_id,
                        section_path=path,
                        text=block_text.strip(),
                        is_table=True,
                        token_count=len(tokenize(block_text)),
                        char_start=base,
                        char_end=base + len(block_text),
                    )
                )
                c_idx += 1
                continue
            offs = tokenize(block_text)
            for c_start, c_end, n_tok in _window_prose(block_text, offs, size, overlap):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}_s{s_idx}_c{c_idx}",
                        doc_id=doc_id,
                        section_path=path,
                        text=block_text[c_start:c_end].strip(),
                        is_table=False,
                        token_count=n_tok,
                        char_start=base + c_start,
                        char_end=base + c_end,
                    )
                )
                c_idx += 1
    return chunks


def pdf_to_markdown(path: Path) -> str:
    """Extract PDF text as pseudo-markdown using a font-size heading heuristic.

    Spans noticeably larger than the document's median font size become
    headings (bigger = higher level). Best-effort: PDFs without size variation
    degrade gracefully to one flat section.
    """
    import fitz  # pymupdf; imported lazily so md/txt ingestion needs no PDF dep loaded

    doc = fitz.open(path)
    sized_lines: list[tuple[float, str]] = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                if text:
                    sized_lines.append((max(s["size"] for s in spans), text))
    doc.close()
    if not sized_lines:
        return ""
    sizes = sorted(s for s, _ in sized_lines)
    median = sizes[len(sizes) // 2]
    out: list[str] = []
    for size, text in sized_lines:
        if size >= median * 1.35 and len(text) < 120:
            out.append(f"# {text}")
        elif size >= median * 1.15 and len(text) < 120:
            out.append(f"## {text}")
        else:
            out.append(text)
    return "\n".join(out)


def chunk_file(path: Path, tokenize: Offsets, size: int, overlap: int) -> list[Chunk]:
    """Chunk a .md/.txt/.pdf file. doc_id = slugified filename stem (stable)."""
    doc_id = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    if path.suffix.lower() == ".pdf":
        text = pdf_to_markdown(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    return chunk_text(doc_id, text, tokenize, size, overlap)
