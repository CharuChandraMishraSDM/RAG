from pydantic import BaseModel, ConfigDict, Field


class PageText(BaseModel):
    """Raw text of one PDF page. Page numbers are 1-based, as users see them."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    text: str


class ExtractedDocument(BaseModel):
    """Output of PDF extraction and input to chunking."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    filename: str
    pages: list[PageText]

    @property
    def page_count(self) -> int:
        return len(self.pages)


class DocumentChunk(BaseModel):
    """A retrievable unit of text with the metadata needed to cite its source.

    `chunk_index` is unique within a document and increases in reading order
    across pages, so `(document_id, chunk_index)` identifies a chunk.
    """

    model_config = ConfigDict(frozen=True)

    document_id: str
    filename: str
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    embedding_dimension: int
