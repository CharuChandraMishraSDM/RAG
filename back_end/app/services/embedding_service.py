"""Generate L2-normalized sentence embeddings for chunks and queries."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.exceptions import EmbeddingError
from app.schemas.document import DocumentChunk

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_ZERO_NORM_EPSILON = 1e-12


@dataclass(frozen=True)
class ChunkEmbeddings:
    """Chunks paired with their vectors: `vectors[i]` embeds `chunks[i]`.

    Keeping both in one object means metadata cannot drift out of alignment
    with the vectors when they are handed to the vector store.
    """

    chunks: list[DocumentChunk]
    vectors: np.ndarray

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.chunks):
            raise ValueError(
                f"Expected {len(self.chunks)} vectors, got array of shape {self.vectors.shape}."
            )


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, so inner product equals cosine similarity.

    Zero vectors are left as zeros instead of producing NaNs.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, _ZERO_NORM_EPSILON)


def _validate_texts(texts: Sequence[str]) -> None:
    if isinstance(texts, str):
        raise TypeError("texts must be a sequence of strings, not a single string.")
    if len(texts) == 0:
        raise ValueError("texts must contain at least one string.")
    for position, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"texts[{position}] must be a non-empty string.")


class EmbeddingService:
    """Wraps a Sentence Transformers model loaded once at construction.

    Create one instance per process and reuse it: loading the model takes
    seconds, while encoding a batch of chunks takes milliseconds.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = self._load_model(model_name, device)
        self.dimension = self._read_dimension()

    @staticmethod
    def _load_model(model_name: str, device: str) -> SentenceTransformer:
        logger.info("Loading embedding model %s on %s", model_name, device)
        try:
            return SentenceTransformer(model_name, device=device)
        except Exception as exc:  # network, missing files, bad name, bad device
            raise EmbeddingError(f"Could not load embedding model '{model_name}': {exc}") from exc

    def _read_dimension(self) -> int:
        """Ask the model for its output size instead of assuming 384."""
        dimension = self._model.get_embedding_dimension()
        if not dimension:
            raise EmbeddingError(f"Model '{self.model_name}' does not report an embedding dimension.")
        return int(dimension)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts in batches. Returns float32 of shape (len(texts), dimension)."""
        _validate_texts(texts)
        try:
            raw = self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(f"Embedding inference failed: {exc}") from exc

        vectors = np.asarray(raw, dtype=np.float32)
        expected_shape = (len(texts), self.dimension)
        if vectors.shape != expected_shape:
            raise EmbeddingError(
                f"Model returned shape {vectors.shape}, expected {expected_shape}."
            )
        return np.ascontiguousarray(l2_normalize(vectors))

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one search query. Returns float32 of shape (dimension,)."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        return self.embed_texts([query])[0]

    def embed_chunks(self, chunks: Sequence[DocumentChunk]) -> ChunkEmbeddings:
        """Embed chunk texts while keeping each vector paired with its metadata."""
        chunk_list = list(chunks)
        vectors = self.embed_texts([chunk.text for chunk in chunk_list])
        return ChunkEmbeddings(chunks=chunk_list, vectors=vectors)
