class DocuMindError(Exception):
    """Base class for errors raised by DocuMind services."""


class InvalidDocumentError(DocuMindError):
    """The uploaded file is not a PDF we can process."""


class EmbeddingError(DocuMindError):
    """The embedding model could not be loaded or failed during inference."""
