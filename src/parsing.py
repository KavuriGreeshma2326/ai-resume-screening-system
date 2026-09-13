"""
Turns an uploaded resume file into plain text.

Supports .pdf, .docx and .txt. Everything downstream works on the plain
text this module returns, so adding a new file type means adding one
branch here and nothing else.
"""

from __future__ import annotations

import io
import re

import pdfplumber
from docx import Document

SUPPORTED_TYPES = ("pdf", "docx", "txt")


class UnsupportedFileType(Exception):
    """Raised when a file extension has no reader."""


def _read_pdf(data: bytes) -> str:
    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _read_docx(data: bytes) -> str:
    document = Document(io.BytesIO(data))
    parts = [para.text for para in document.paragraphs]
    # Many resumes lay out skills inside tables, which paragraphs miss.
    for table in document.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


def _read_txt(data: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def clean_text(text: str) -> str:
    """Normalise whitespace and strip characters that confuse the matcher."""
    text = text.replace("\u2022", " ")          # bullet
    text = text.replace("\xa0", " ")            # non-breaking space
    text = re.sub(r"[\u2010-\u2015]", "-", text)  # various dashes -> hyphen
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(filename: str, data: bytes) -> str:
    """
    Read `data` (raw file bytes) according to the extension of `filename`.

    Raises UnsupportedFileType for anything other than pdf/docx/txt.
    """
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension == "pdf":
        raw = _read_pdf(data)
    elif extension == "docx":
        raw = _read_docx(data)
    elif extension == "txt":
        raw = _read_txt(data)
    else:
        raise UnsupportedFileType(
            f"'{filename}' is a .{extension or 'unknown'} file. "
            f"Upload a PDF, DOCX or TXT instead."
        )

    return clean_text(raw)
