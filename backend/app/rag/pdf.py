"""PDF extraction: pages in, structured blocks out.

Role in architecture: the previous implementation flattened a PDF into one
markdown string using only a font-size heading heuristic. That lost page
numbers (so citations could not say "page 3"), mangled two-column layouts by
reading straight across the fold, repeated every running header/footer as if
it were body text, and — worst — returned "" for an encrypted or scanned PDF,
which the caller then happily indexed as a successful empty document.

This module extracts page-by-page with explicit, typed failure modes so the
document lifecycle can record WHY a document could not be ingested instead of
reporting a silent success.

Nothing here calls an LLM: extraction is deterministic, so the same PDF always
produces the same chunks and the same citations.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.models.schemas import DocumentErrorCode


class PdfExtractionError(Exception):
    """Extraction failed in a way the user needs to know about."""

    def __init__(self, code: DocumentErrorCode, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class PageBlock:
    """One structural unit on one page."""

    text: str
    page_number: int  # 1-based
    is_table: bool = False
    is_heading: bool = False
    heading_level: int = 0


@dataclass
class ExtractedDocument:
    blocks: list[PageBlock] = field(default_factory=list)
    page_count: int = 0

    @property
    def total_chars(self) -> int:
        return sum(len(b.text) for b in self.blocks)


_WS_RE = re.compile(r"[ \t]+")
# A line that is only digits, or "Page 3", or "3 of 12" — page furniture, not content.
_PAGE_NUM_RE = re.compile(r"^\s*(?:page\s+)?\d+\s*(?:/|of\s+)?\s*\d*\s*$", re.I)


def _normalize_line(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _detect_repeated_furniture(pages_lines: list[list[str]]) -> set[str]:
    """Find running headers/footers: short lines that recur near the top or
    bottom edge of most pages.

    Without this, a 20-page policy PDF repeats its title and page footer 20
    times in the index, and those near-duplicate chunks crowd out real content
    in retrieval — the classic "every result is the document header" failure.
    Requires >= 3 pages: on a 2-page document a legitimately repeated sentence
    is not evidence of furniture.
    """
    if len(pages_lines) < 3:
        return set()

    edge_counts: Counter[str] = Counter()
    for lines in pages_lines:
        if not lines:
            continue
        # only the outer few lines of each page can be furniture
        candidates = lines[:2] + lines[-2:]
        for line in {c for c in candidates if c}:
            if len(line) <= 120:  # a long paragraph is not a running header
                edge_counts[line] += 1

    threshold = max(3, int(len(pages_lines) * 0.6))
    return {line for line, count in edge_counts.items() if count >= threshold}


def _sort_blocks_reading_order(
    raw_blocks: list[dict[str, Any]], page_width: float
) -> list[dict[str, Any]]:
    """Order blocks top-to-bottom, and for multi-column layouts, column by column.

    PyMuPDF returns blocks in internal document order, which for a two-column
    layout interleaves the columns — reading straight across the fold and
    producing sentences spliced from unrelated columns. We detect a column
    split by checking whether blocks cluster on either side of the page's
    horizontal midpoint with few blocks straddling it, and if so, emit the
    whole left column before the whole right column.
    """
    if not raw_blocks:
        return []

    mid = page_width / 2.0
    left, right, straddling = [], [], []
    for b in raw_blocks:
        x0, x1 = b["bbox"][0], b["bbox"][2]
        if x1 < mid + page_width * 0.02:
            left.append(b)
        elif x0 > mid - page_width * 0.02:
            right.append(b)
        else:
            straddling.append(b)

    # Treat as two-column only when both sides are substantially populated and
    # few blocks span the fold (a full-width title over two columns is fine).
    is_two_column = (
        len(left) >= 2
        and len(right) >= 2
        and len(straddling) <= max(1, (len(left) + len(right)) // 8)
    )

    def by_vertical(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(blocks, key=lambda b: (round(b["bbox"][1], 1), round(b["bbox"][0], 1)))

    if not is_two_column:
        return by_vertical(raw_blocks)
    # full-width blocks (titles) first, then left column, then right
    return by_vertical(straddling) + by_vertical(left) + by_vertical(right)


def _extract_tables(page: Any) -> list[tuple[tuple[float, float, float, float], str]]:
    """Return [(bbox, markdown)] for tables PyMuPDF can find on this page.

    Tables are emitted as markdown pipe rows so the downstream chunker's
    existing table handling keeps them atomic, and so an LLM reading the chunk
    sees an actual grid rather than numbers that lost their column headers.
    """
    found: list[tuple[tuple[float, float, float, float], str]] = []
    try:
        tables = page.find_tables()
    except Exception:  # noqa: BLE001 - table finding is best-effort by design
        return found

    for table in getattr(tables, "tables", []) or []:
        try:
            rows = table.extract()
        except Exception:  # noqa: BLE001
            continue
        cleaned = [
            [_normalize_line(str(cell)) if cell is not None else "" for cell in row]
            for row in rows
            if row is not None
        ]
        cleaned = [r for r in cleaned if any(c for c in r)]
        if len(cleaned) < 2:  # a single row is not a table worth preserving
            continue
        width = max(len(r) for r in cleaned)
        lines = []
        for i, row in enumerate(cleaned):
            padded = row + [""] * (width - len(row))
            lines.append("| " + " | ".join(padded) + " |")
            if i == 0:
                lines.append("|" + "---|" * width)
        found.append((tuple(table.bbox), "\n".join(lines)))
    return found


def _bbox_overlaps(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def extract_pdf(path: Path) -> ExtractedDocument:
    """Extract a PDF into page-tagged blocks, or raise PdfExtractionError.

    Raises rather than returning empty so a caller can never mistake an
    unreadable document for an empty-but-successful one.
    """
    import fitz  # pymupdf; imported lazily so md/txt ingestion needs no PDF dep

    settings = get_settings()

    try:
        doc = fitz.open(path)
    except Exception as exc:  # noqa: BLE001 - any fitz failure means unusable input
        raise PdfExtractionError(
            DocumentErrorCode.CORRUPT_DOCUMENT, f"could not open PDF: {exc}"
        ) from exc

    try:
        if doc.needs_pass or doc.is_encrypted:
            raise PdfExtractionError(
                DocumentErrorCode.ENCRYPTED_DOCUMENT,
                "PDF is password-protected; upload an unprotected copy",
            )

        page_count = doc.page_count
        if page_count == 0:
            raise PdfExtractionError(DocumentErrorCode.EMPTY_DOCUMENT, "PDF has no pages")
        if page_count > settings.max_pdf_pages:
            raise PdfExtractionError(
                DocumentErrorCode.TOO_MANY_PAGES,
                f"PDF has {page_count} pages; the limit is {settings.max_pdf_pages}",
            )

        # ---- pass 1: raw lines per page (for furniture detection) + sizes
        per_page_lines: list[list[str]] = []
        per_page_payload: list[list[dict[str, Any]]] = []
        all_sizes: list[float] = []

        for page in doc:
            try:
                page_dict = page.get_text("dict")
            except Exception:  # noqa: BLE001 - skip an unreadable page, keep the rest
                per_page_lines.append([])
                per_page_payload.append([])
                continue

            raw_blocks = [b for b in page_dict.get("blocks", []) if b.get("lines")]
            ordered = _sort_blocks_reading_order(raw_blocks, page_dict.get("width") or 612.0)

            payload: list[dict[str, Any]] = []
            lines_on_page: list[str] = []
            for block in ordered:
                block_lines = []
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    text = _normalize_line("".join(s.get("text", "") for s in spans))
                    if not text:
                        continue
                    size = max((s.get("size", 0.0) for s in spans), default=0.0)
                    all_sizes.append(size)
                    block_lines.append({"text": text, "size": size})
                    lines_on_page.append(text)
                if block_lines:
                    payload.append({"bbox": tuple(block["bbox"]), "lines": block_lines})
            per_page_lines.append(lines_on_page)
            per_page_payload.append(payload)

        furniture = _detect_repeated_furniture(per_page_lines)
        median_size = sorted(all_sizes)[len(all_sizes) // 2] if all_sizes else 0.0

        # ---- pass 2: build blocks, splicing tables in at their page position
        blocks: list[PageBlock] = []
        for page_index, payload in enumerate(per_page_payload):
            page_number = page_index + 1
            tables = _extract_tables(doc[page_index]) if page_index < doc.page_count else []
            table_boxes = [bbox for bbox, _ in tables]

            for _bbox, markdown in tables:
                blocks.append(
                    PageBlock(text=markdown, page_number=page_number, is_table=True)
                )

            for block in payload:
                # text already captured inside a table region would duplicate it
                if any(_bbox_overlaps(block["bbox"], tb) for tb in table_boxes):
                    continue

                kept = [
                    ln
                    for ln in block["lines"]
                    if ln["text"] not in furniture and not _PAGE_NUM_RE.match(ln["text"])
                ]
                if not kept:
                    continue

                block_text = "\n".join(ln["text"] for ln in kept)
                max_size = max(ln["size"] for ln in kept)
                is_heading = (
                    median_size > 0
                    and max_size >= median_size * 1.15
                    and len(block_text) < 120
                    and len(kept) <= 2
                )
                level = 1 if (median_size and max_size >= median_size * 1.35) else 2
                blocks.append(
                    PageBlock(
                        text=block_text,
                        page_number=page_number,
                        is_heading=is_heading,
                        heading_level=level if is_heading else 0,
                    )
                )

        extracted = ExtractedDocument(blocks=blocks, page_count=page_count)

        # ---- scanned/image-only detection
        # A PDF of page images extracts (almost) no characters. Indexing that as
        # a successful empty document is the silent-failure mode this guards.
        if extracted.total_chars < settings.min_chars_per_page_for_text_pdf * page_count:
            if extracted.total_chars == 0:
                raise PdfExtractionError(
                    DocumentErrorCode.UNSUPPORTED_SCANNED_DOCUMENT,
                    "no extractable text — this looks like a scanned/image-only PDF. "
                    "OCR is not supported; upload a text-based PDF.",
                )
            raise PdfExtractionError(
                DocumentErrorCode.NO_EXTRACTABLE_TEXT,
                f"only {extracted.total_chars} extractable characters across "
                f"{page_count} page(s) — too little to index reliably.",
            )

        return extracted
    finally:
        doc.close()


def blocks_to_markdown(extracted: ExtractedDocument) -> str:
    """Render extracted blocks as markdown with page markers.

    The chunker consumes markdown; the page markers let it tag each chunk with
    the page it came from so citations can name a page.
    """
    out: list[str] = []
    current_page = None
    for block in extracted.blocks:
        if block.page_number != current_page:
            out.append(f"{PAGE_MARKER_PREFIX}{block.page_number}{PAGE_MARKER_SUFFIX}")
            current_page = block.page_number
        if block.is_heading:
            out.append(f"{'#' * max(1, block.heading_level)} {block.text}")
        else:
            out.append(block.text)
    return "\n\n".join(out)


# Sentinel markers the chunker strips out after using them to assign page
# numbers. Chosen to be something no real document line would contain.
PAGE_MARKER_PREFIX = "<<<SENTINEL_PAGE:"
PAGE_MARKER_SUFFIX = ">>>"
PAGE_MARKER_RE = re.compile(
    re.escape(PAGE_MARKER_PREFIX) + r"(\d+)" + re.escape(PAGE_MARKER_SUFFIX)
)
