import pytest

from app.schemas.document import ExtractedDocument, PageText
from app.services.chunking_service import ChunkingService, normalize_text


def make_document(*page_texts: str) -> ExtractedDocument:
    return ExtractedDocument(
        document_id="doc_001",
        filename="database_runbook.pdf",
        pages=[PageText(page_number=i, text=t) for i, t in enumerate(page_texts, start=1)],
    )


def numbered_words(count: int) -> str:
    """Unique tokens make it possible to see exactly which text two chunks share."""
    return " ".join(f"w{i:04d}" for i in range(count))


def shared_boundary_length(previous: str, following: str) -> int:
    """Length of the longest suffix of `previous` that is a prefix of `following`."""
    for size in range(min(len(previous), len(following)), 0, -1):
        if previous.endswith(following[:size]):
            return size
    return 0


def test_short_single_page_produces_one_chunk_with_full_metadata():
    chunks = ChunkingService().chunk_document(make_document("Connection pool size is 20."))

    assert len(chunks) == 1
    assert chunks[0].model_dump() == {
        "document_id": "doc_001",
        "filename": "database_runbook.pdf",
        "page_number": 1,
        "chunk_index": 0,
        "text": "Connection pool size is 20.",
    }


def test_long_page_is_split_into_chunks_within_size_limit():
    service = ChunkingService(chunk_size=200, chunk_overlap=40)

    chunks = service.chunk_document(make_document(numbered_words(300)))

    assert len(chunks) > 1
    assert all(0 < len(chunk.text) <= 200 for chunk in chunks)


def test_consecutive_chunks_overlap_by_at_most_chunk_overlap():
    service = ChunkingService(chunk_size=200, chunk_overlap=40)

    texts = service.split_text(numbered_words(300))

    for previous, following in zip(texts, texts[1:]):
        assert 0 < shared_boundary_length(previous, following) <= 40


def test_zero_overlap_produces_disjoint_chunks():
    texts = ChunkingService(chunk_size=200, chunk_overlap=0).split_text(numbered_words(300))

    for previous, following in zip(texts, texts[1:]):
        assert shared_boundary_length(previous, following) == 0


def test_chunks_never_span_pages_and_keep_their_page_number():
    service = ChunkingService(chunk_size=200, chunk_overlap=40)
    document = make_document("alpha " * 60, "beta " * 60, "gamma " * 60)

    chunks = service.chunk_document(document)

    for chunk in chunks:
        expected_word = {1: "alpha", 2: "beta", 3: "gamma"}[chunk.page_number]
        assert set(chunk.text.split()) == {expected_word}
    assert {chunk.page_number for chunk in chunks} == {1, 2, 3}


def test_chunk_index_is_sequential_across_pages():
    service = ChunkingService(chunk_size=200, chunk_overlap=40)

    chunks = service.chunk_document(make_document("alpha " * 60, "beta " * 60))

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.document_id == "doc_001" for chunk in chunks)
    assert all(chunk.filename == "database_runbook.pdf" for chunk in chunks)


def test_blank_and_whitespace_pages_produce_no_chunks_but_keep_page_numbers():
    chunks = ChunkingService().chunk_document(make_document("", "  \n\t \x00 ", "Real content."))

    assert len(chunks) == 1
    assert chunks[0].page_number == 3
    assert chunks[0].chunk_index == 0


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t\r\n", "\x00\x0c"])
def test_split_text_returns_empty_list_for_text_without_content(text):
    assert ChunkingService().split_text(text) == []


def test_split_text_rejects_non_string_input():
    with pytest.raises(TypeError):
        ChunkingService().split_text(None)  # type: ignore[arg-type]


def test_normalize_text_cleans_pdf_artifacts_but_keeps_paragraphs():
    raw = "Title\r\n\r\n\r\n\r\nFirst\t\tline   here\x00\n   indented line  "

    assert normalize_text(raw) == "Title\n\nFirst line here\nindented line"


def test_prefers_paragraph_boundaries_over_mid_sentence_breaks():
    paragraph_one = "Backups run nightly at 02:00. " * 4
    paragraph_two = "Restores require approval from the on-call DBA. " * 4
    service = ChunkingService(chunk_size=250, chunk_overlap=0)

    texts = service.split_text(f"{paragraph_one}\n\n{paragraph_two}")

    assert texts == [paragraph_one.strip(), paragraph_two.strip()]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (-5, 0), (100, -1), (100, 100), (100, 150)],
)
def test_invalid_configuration_is_rejected(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        ChunkingService(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_defaults_are_800_and_150():
    service = ChunkingService()

    assert (service.chunk_size, service.chunk_overlap) == (800, 150)
