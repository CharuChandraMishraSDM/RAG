from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import documents
from app.core.exceptions import EmbeddingError, InvalidDocumentError

app = FastAPI(
    title="DocuMind API",
    description="Enterprise Document Intelligence using RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)


@app.exception_handler(InvalidDocumentError)
async def invalid_document_handler(request: Request, exc: InvalidDocumentError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(EmbeddingError)
async def embedding_error_handler(request: Request, exc: EmbeddingError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "application": "DocuMind",
        "version": "1.0.0",
    }


@app.get("/")
async def root():
    return {
        "message": "Welcome to DocuMind API",
        "docs": "/docs",
    }
