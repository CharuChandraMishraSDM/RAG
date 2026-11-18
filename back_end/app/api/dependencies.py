from functools import lru_cache

from app.config import get_settings
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService


@lru_cache
def get_chunking_service() -> ChunkingService:
    settings = get_settings()
    return ChunkingService(settings.chunk_size, settings.chunk_overlap)


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Return the process-wide embedding service, loading the model on first use."""
    settings = get_settings()
    return EmbeddingService(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
