import numpy as np
import pytest

from app.core.exceptions import EmbeddingError
from app.schemas.document import DocumentChunk, ExtractedDocument, PageText
from app.services import embedding_service
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import (
    DEFAULT_EMBEDDING_MODEL,
    ChunkEmbeddings,
    EmbeddingService,
    l2_normalize,
)
from app.tests.conftest import FAKE_DIMENSION


def make_chunk(index: int, text: str, page_number: int = 1) -> DocumentChunk:
    return DocumentChunk(
        document_id="doc_001",
        filename="database_runbook.pdf",
        page_number=page_number,
        chunk_index=index,
        text=text,
    )


def test_batch_embedding_returns_one_unit_vector_per_text(fake_embedding_service):
    vectors = fake_embedding_service.embed_texts(["first", "second", "third"])

    assert vectors.shape == (3, FAKE_DIMENSION)
    assert vectors.dtype == np.float32
    assert vectors.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)


def test_texts_are_sent_to_model_in_order_with_configured_batch_size(fake_embedding_service):
    fake_embedding_service.embed_texts(["a", "b"])

    call = fake_embedding_service._model.encode_calls[-1]
    assert call["texts"] == ["a", "b"]
    assert call["batch_size"] == 8


def test_query_embedding_is_a_single_unit_vector(fake_embedding_service):
    query_vector = fake_embedding_service.embed_query("How large is the pool?")

    assert query_vector.shape == (FAKE_DIMENSION,)
    assert np.linalg.norm(query_vector) == pytest.approx(1.0, rel=1e-5)


def test_query_and_identical_document_text_have_cosine_similarity_one(fake_embedding_service):
    text = "Connection pool size is 20."

    similarity = fake_embedding_service.embed_query(text) @ fake_embedding_service.embed_texts([text])[0]

    assert similarity == pytest.approx(1.0, rel=1e-5)


def test_dimension_is_read_from_the_model(fake_embedding_service):
    assert fake_embedding_service.dimension == FAKE_DIMENSION


def test_embed_chunks_keeps_vectors_aligned_with_metadata(fake_embedding_service):
    chunks = [make_chunk(0, "pool size", page_number=4), make_chunk(1, "backup policy", page_number=7)]

    embedded = fake_embedding_service.embed_chunks(chunks)

    assert embedded.chunks == chunks
    assert embedded.vectors.shape == (2, FAKE_DIMENSION)
    np.testing.assert_allclose(
        embedded.vectors[1], fake_embedding_service.embed_query("backup policy"), rtol=1e-6
    )


@pytest.mark.parametrize(
    ("texts", "error"),
    [
        ([], ValueError),
        (["valid", "   "], ValueError),
        (["valid", ""], ValueError),
        (["valid", None], ValueError),
        ("a bare string", TypeError),
    ],
)
def test_embed_texts_rejects_invalid_input(fake_embedding_service, texts, error):
    with pytest.raises(error):
        fake_embedding_service.embed_texts(texts)


@pytest.mark.parametrize("query", ["", "  \n ", None])
def test_embed_query_rejects_blank_input(fake_embedding_service, query):
    with pytest.raises(ValueError):
        fake_embedding_service.embed_query(query)


def test_inference_failure_is_wrapped_in_embedding_error(fake_embedding_service, monkeypatch):
    def broken_encode(*args, **kwargs):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(fake_embedding_service._model, "encode", broken_encode)

    with pytest.raises(EmbeddingError, match="out of memory"):
        fake_embedding_service.embed_texts(["text"])


def test_unexpected_output_shape_is_reported(fake_embedding_service, monkeypatch):
    monkeypatch.setattr(
        fake_embedding_service._model, "encode", lambda texts, **kwargs: np.ones((len(texts), 10))
    )

    with pytest.raises(EmbeddingError, match="expected"):
        fake_embedding_service.embed_texts(["text"])


def test_model_load_failure_raises_embedding_error(monkeypatch):
    def failing_loader(*args, **kwargs):
        raise OSError("repository not found")

    monkeypatch.setattr(embedding_service, "SentenceTransformer", failing_loader)

    with pytest.raises(EmbeddingError, match="Could not load embedding model 'missing/model'"):
        EmbeddingService(model_name="missing/model")


def test_l2_normalize_produces_unit_rows_and_leaves_zero_vectors_finite():
    vectors = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

    normalized = l2_normalize(vectors)

    np.testing.assert_allclose(normalized[0], [0.6, 0.8])
    np.testing.assert_array_equal(normalized[1], [0.0, 0.0])


def test_chunk_embeddings_rejects_misaligned_vectors():
    with pytest.raises(ValueError):
        ChunkEmbeddings(chunks=[make_chunk(0, "only one")], vectors=np.zeros((2, 4)))


# The tests below load the real model (~90 MB, downloaded once into the
# Hugging Face cache). They are skipped by default; run with `pytest -m integration`.


@pytest.fixture(scope="module")
def real_embedding_service() -> EmbeddingService:
    return EmbeddingService(model_name=DEFAULT_EMBEDDING_MODEL, device="cpu")


@pytest.mark.integration
def test_real_model_produces_384_dimensional_unit_vectors(real_embedding_service):
    vectors = real_embedding_service.embed_texts(["Database connection pool", "Quarterly revenue"])

    assert real_embedding_service.dimension == 384
    assert vectors.shape == (2, 384)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)


@pytest.mark.integration
def test_real_model_ranks_relevant_chunk_above_unrelated_chunk(real_embedding_service):
    chunks = [
        make_chunk(0, "Set max_connections in the database pool configuration to 50."),
        make_chunk(1, "The cafeteria serves lunch between noon and two o'clock."),
    ]

    embedded = real_embedding_service.embed_chunks(chunks)
    query = real_embedding_service.embed_query("How do I configure the connection pool size?")
    similarities = embedded.vectors @ query

    assert similarities[0] > similarities[1]


@pytest.mark.integration
def test_real_pipeline_from_pdf_to_embeddings(real_embedding_service, make_pdf):
    from app.services.pdf_service import extract_document

    pdf = make_pdf(["Pool size is 20 connections.", "", "Backups run nightly."])
    document = extract_document("runbook.pdf", pdf, max_size_bytes=10_000_000)

    embedded = real_embedding_service.embed_chunks(ChunkingService().chunk_document(document))

    assert [chunk.page_number for chunk in embedded.chunks] == [1, 3]
    assert embedded.vectors.shape == (2, 384)
