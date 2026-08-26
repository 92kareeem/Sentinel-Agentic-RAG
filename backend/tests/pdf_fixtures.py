"""Deterministic PDF fixtures for ingestion/retrieval tests.

Built programmatically rather than checked in as binaries so the corpus is
reviewable in diffs, and so each fixture's known facts live next to the bytes
that contain them — a test can assert "the answer is on page 3" against a
document whose page 3 content is visible right here.

Requires pymupdf (already a runtime dependency); no network access.
"""

from __future__ import annotations

from pathlib import Path

# Facts deliberately placed at known locations so tests can assert retrieval
# reached the right page/section rather than merely returning something.
FACT_PAGE_1 = "The reimbursement window is 45 days from the purchase date."
FACT_PAGE_2 = "Production database access requires the on-call certification exam OPS-204."
FACT_PAGE_3 = "Unused allowance does not roll over between refresh cycles."
TABLE_FACT_ROLE = "Designer"
TABLE_FACT_BUDGET = "$2,800"
ABSENT_TOPIC = "parental leave"


def _new_doc():
    import fitz

    return fitz.open()


def _prepare(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_page(doc, lines: list[tuple[str, float]], *, header: str | None = None) -> None:
    """Add a page; lines are (text, fontsize). Larger sizes read as headings."""
    page = doc.new_page()
    y = 60.0
    if header is not None:
        page.insert_text((60, 36), header, fontsize=8)  # running header
    for text, size in lines:
        page.insert_text((60, y), text, fontsize=size)
        y += size + 10
    page.insert_text((300, 760), f"{doc.page_count}", fontsize=8)  # page number footer


def simple_text_pdf(path: Path) -> Path:
    path = _prepare(path)
    doc = _new_doc()
    _write_page(
        doc,
        [
            ("Expense Policy", 20),
            ("Reimbursement", 15),
            (FACT_PAGE_1, 11),
            ("Receipts must be attached for any claim above fifty dollars.", 11),
        ],
    )
    doc.save(path)
    doc.close()
    return path


def multipage_pdf(path: Path) -> Path:
    """3 pages, repeated header/footer, a heading per page, one table."""
    path = _prepare(path)
    doc = _new_doc()
    header = "Meridian Robotics - Internal Handbook"

    _write_page(
        doc,
        [
            ("Employee Handbook", 20),
            ("Reimbursement", 15),
            (FACT_PAGE_1, 11),
            ("Receipts must be attached for any claim above fifty dollars.", 11),
        ],
        header=header,
    )
    _write_page(
        doc,
        [
            ("Access Requests", 15),
            ("System access follows least privilege.", 11),
            (FACT_PAGE_2, 11),
            ("Requests go through the ACCESS-REQ form and require manager approval.", 11),
        ],
        header=header,
    )
    _write_page(
        doc,
        [
            ("Equipment Allowances", 15),
            ("| Role | Laptop budget | Refresh cycle |", 10),
            ("| Engineer | $2,400 | 3 years |", 10),
            (f"| {TABLE_FACT_ROLE} | {TABLE_FACT_BUDGET} | 3 years |", 10),
            ("| Sales | $1,600 | 4 years |", 10),
            (FACT_PAGE_3, 11),
        ],
        header=header,
    )
    doc.save(path)
    doc.close()
    return path


def two_column_pdf(path: Path) -> Path:
    """Two columns; reading straight across would splice unrelated sentences."""
    path = _prepare(path)
    import fitz

    doc = _new_doc()
    page = doc.new_page()
    left = (
        "The left column discusses shipping timelines in detail and "
        "continues for several lines without interruption."
    )
    right = (
        "The right column covers warranty terms separately and is not "
        "a continuation of the shipping discussion."
    )
    page.insert_textbox(fitz.Rect(50, 60, 280, 700), left, fontsize=11)
    page.insert_textbox(fitz.Rect(310, 60, 545, 700), right, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def empty_pdf(path: Path) -> Path:
    """A structurally valid PDF with a page but no text — the scanned-PDF shape."""
    path = _prepare(path)
    doc = _new_doc()
    doc.new_page()
    doc.save(path)
    doc.close()
    return path


def encrypted_pdf(path: Path) -> Path:
    path = _prepare(path)
    import fitz

    doc = _new_doc()
    _write_page(doc, [("Secret", 20), ("classified content", 11)])
    doc.save(path, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    doc.close()
    return path


def corrupt_pdf(path: Path) -> Path:
    path = _prepare(path)
    path.write_bytes(b"%PDF-1.7\nnot actually a pdf at all\n%%EOF")
    return path


def unicode_pdf(path: Path) -> Path:
    path = _prepare(path)
    doc = _new_doc()
    _write_page(
        doc,
        [
            ("Ubersicht", 20),  # ASCII-safe: base14 fonts lack full Unicode coverage
            ("Cost: 1.234,56 EUR per unit, rounded to 2 decimals.", 11),
            ("Ratio is 5% +/- 0.5% across regions.", 11),
        ],
    )
    doc.save(path)
    doc.close()
    return path
