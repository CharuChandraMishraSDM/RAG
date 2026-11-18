"""Validate uploaded PDFs and extract their text page by page."""

import uuid
from pathlib import PurePath

import pymupdf

from app.core.exceptions import InvalidDocumentError
from app.schemas.document import ExtractedDocument, PageText

PDF_MAGIC_BYTES = b"%PDF-"


def validate_pdf_upload(filename: str, content: bytes, max_size_bytes: int) -> None:
    """Reject files that are obviously not processable before parsing them.

    The extension check gives users a clear message; the magic-byte check
    catches renamed non-PDF files. Full structural validation happens when
    PyMuPDF opens the file.
    """
    if PurePath(filename).suffix.lower() != ".pdf":
        raise InvalidDocumentError(f"'{filename}' is not a .pdf file.")
    if not content:
        raise InvalidDocumentError(f"'{filename}' is empty.")
    if len(content) > max_size_bytes:
        raise InvalidDocumentError(
            f"'{filename}' exceeds the maximum upload size of {max_size_bytes} bytes."
        )
    if not content.startswith(PDF_MAGIC_BYTES):
        raise InvalidDocumentError(f"'{filename}' does not have a valid PDF header.")


def extract_pages(content: bytes) -> list[PageText]:
    """Return the text of every page, including pages with no text.

    Blank pages are kept so page numbers always match the original PDF;
    chunking is responsible for skipping pages without content.
    """
    try:
        with pymupdf.open(stream=content, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise InvalidDocumentError("Password-protected PDFs are not supported.")
            if pdf.page_count == 0:
                raise InvalidDocumentError("The PDF contains no pages.")
            return [
                PageText(page_number=page.number + 1, text=page.get_text("text"))
                for page in pdf
            ]
    except (pymupdf.FileDataError, pymupdf.EmptyFileError) as exc:
        raise InvalidDocumentError("The PDF is corrupted or cannot be parsed.") from exc


def extract_document(filename: str, content: bytes, max_size_bytes: int) -> ExtractedDocument:
    validate_pdf_upload(filename, content, max_size_bytes)
    return ExtractedDocument(
        document_id=f"doc_{uuid.uuid4().hex}",
        filename=PurePath(filename).name,
        pages=extract_pages(content),
    )
