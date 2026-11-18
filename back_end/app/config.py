from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and `.env`.

    Only settings used by implemented features are declared here; unknown
    variables (e.g. LLM keys for later milestones) are ignored rather than
    rejected so a complete `.env` file never prevents startup.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "DocuMind"
    debug: bool = False

    max_upload_size_mb: int = Field(default=20, gt=0)

    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=32, gt=0)

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
