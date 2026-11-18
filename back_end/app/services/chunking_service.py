"""Split extracted document pages into overlapping, citable text chunks."""

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.schemas.document import DocumentChunk, ExtractedDocument

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150

# Tried in order: paragraphs, lines, sentences, words, then characters as a
# last resort, so chunks break at the most natural boundary that fits.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Clean PDF-extracted text without destroying its paragraph structure.

    Newlines are kept (collapsed to at most one blank line) because the
    splitter uses them as preferred break points; all other whitespace runs,
    including tabs and non-breaking spaces, become a single space.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARACTERS.sub("", text)
    text = _HORIZONTAL_WHITESPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _EXCESS_BLANK_LINES.sub("\n\n", text).strip()


class ChunkingService:
    """Splits documents into chunks of at most `chunk_size` characters.

    Each page is split independently, so a chunk never spans two pages and
    every chunk can be cited with a single, exact page number. The cost is
    that a sentence running across a page break is split in two.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=SEPARATORS,
            keep_separator="end",  # keep a sentence's period with the sentence
            strip_whitespace=True,
        )

    def split_text(self, text: str) -> list[str]:
        """Split one block of text into non-empty chunks."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}.")
        normalized = normalize_text(text)
        if not normalized:
            return []
        return [chunk for chunk in self._splitter.split_text(normalized) if chunk.strip()]

    def chunk_document(self, document: ExtractedDocument) -> list[DocumentChunk]:
        """Chunk every page of a document, numbering chunks in reading order."""
        chunks: list[DocumentChunk] = []
        for page in document.pages:
            for text in self.split_text(page.text):
                chunks.append(
                    DocumentChunk(
                        document_id=document.document_id,
                        filename=document.filename,
                        page_number=page.page_number,
                        chunk_index=len(chunks),
                        text=text,
                    )
                )
        return chunks
