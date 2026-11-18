import pymupdf
import pytest

from app.core.exceptions import InvalidDocumentError
from app.services.pdf_service import extract_document, extract_pages, validate_pdf_upload

MAX_BYTES = 1024 * 1024


def test_extracts_text_with_one_based_page_numbers(make_pdf):
    pages = extract_pages(make_pdf(["First page text.", "Second page text."]))

    assert [page.page_number for page in pages] == [1, 2]
    assert "First page text." in pages[0].text
    assert "Second page text." in pages[1].text


def test_blank_pages_are_kept_so_page_numbers_stay_accurate(make_pdf):
    pages = extract_pages(make_pdf(["Intro", "", "Appendix"]))

    assert [page.page_number for page in pages] == [1, 2, 3]
    assert pages[1].text.strip() == ""
    assert "Appendix" in pages[2].text


def test_extract_document_assigns_id_and_strips_client_path(make_pdf):
    document = extract_document("reports/runbook.pdf", make_pdf(["Hello"]), MAX_BYTES)

    assert document.document_id.startswith("doc_")
    assert document.filename == "runbook.pdf"
    assert document.page_count == 1


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("notes.txt", b"%PDF-1.7", "not a .pdf"),
        ("empty.pdf", b"", "is empty"),
        ("fake.pdf", b"PK\x03\x04 zip data", "valid PDF header"),
        ("huge.pdf", b"%PDF-" + b"0" * MAX_BYTES, "maximum upload size"),
    ],
)
def test_validate_rejects_invalid_uploads(filename, content, message):
    with pytest.raises(InvalidDocumentError, match=message):
        validate_pdf_upload(filename, content, MAX_BYTES)


def test_corrupted_pdf_raises_invalid_document():
    with pytest.raises(InvalidDocumentError, match="corrupted"):
        extract_pages(b"%PDF-1.7\nthis is not a real pdf body")


def test_password_protected_pdf_is_rejected(make_pdf):
    with pymupdf.open(stream=make_pdf(["secret"]), filetype="pdf") as pdf:
        encrypted = pdf.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user"
        )

    with pytest.raises(InvalidDocumentError, match="Password-protected"):
        extract_pages(encrypted)
