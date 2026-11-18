import hashlib
from collections.abc import Callable, Sequence

import numpy as np
import pymupdf
import pytest

from app.services import embedding_service
from app.services.embedding_service import EmbeddingService

FAKE_DIMENSION = 384


class FakeSentenceTransformer:
    """Stands in for SentenceTransformer so unit tests need no model download.

    Vectors are deterministic per text and deliberately not unit length, so
    tests prove that EmbeddingService performs the normalization itself.
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self.encode_calls: list[dict] = []

    def get_embedding_dimension(self) -> int:
        return FAKE_DIMENSION

    def encode(self, texts: Sequence[str], **kwargs) -> np.ndarray:
        self.encode_calls.append({"texts": list(texts), **kwargs})
        return np.stack([self._vector_for(text) for text in texts])

    @staticmethod
    def _vector_for(text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        return np.random.default_rng(seed).normal(scale=5.0, size=FAKE_DIMENSION)


@pytest.fixture
def fake_embedding_service(monkeypatch: pytest.MonkeyPatch) -> EmbeddingService:
    monkeypatch.setattr(embedding_service, "SentenceTransformer", FakeSentenceTransformer)
    return EmbeddingService(model_name="fake-model", batch_size=8)


@pytest.fixture
def make_pdf() -> Callable[[list[str]], bytes]:
    """Build an in-memory PDF with one page per string ("" gives a blank page)."""

    def _make_pdf(page_texts: list[str]) -> bytes:
        with pymupdf.open() as pdf:
            for text in page_texts:
                page = pdf.new_page()
                if text:
                    page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=10)
            return pdf.tobytes()

    return _make_pdf
