from typing import Annotated

from fastapi import APIRouter, Depends, UploadFile

from app.api.dependencies import get_chunking_service, get_embedding_service
from app.config import Settings, get_settings
from app.core.exceptions import InvalidDocumentError
from app.schemas.document import DocumentUploadResponse
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.pdf_service import extract_document

router = APIRouter(prefix="/documents", tags=["documents"])


def _read_upload(file: UploadFile, max_size_bytes: int) -> bytes:
    """Read at most one byte past the limit so oversized files are never fully buffered."""
    return file.file.read(max_size_bytes + 1)


# A sync route: FastAPI runs it in a worker thread, so CPU-bound parsing and
# embedding do not block the event loop.
@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
def upload_document(
    file: UploadFile,
    settings: Annotated[Settings, Depends(get_settings)],
    chunker: Annotated[ChunkingService, Depends(get_chunking_service)],
    embedder: Annotated[EmbeddingService, Depends(get_embedding_service)],
) -> DocumentUploadResponse:
    content = _read_upload(file, settings.max_upload_size_bytes)
    document = extract_document(file.filename or "", content, settings.max_upload_size_bytes)

    chunks = chunker.chunk_document(document)
    if not chunks:
        raise InvalidDocumentError(
            f"'{document.filename}' contains no extractable text. Scanned PDFs are not supported."
        )
    # Embeddings are not persisted yet; the vector store milestone stores them here.
    embedded = embedder.embed_chunks(chunks)

    return DocumentUploadResponse(
        document_id=document.document_id,
        filename=document.filename,
        page_count=document.page_count,
        chunk_count=len(embedded.chunks),
        embedding_dimension=embedded.vectors.shape[1],
    )
