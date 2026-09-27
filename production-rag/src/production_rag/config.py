from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")
    db_path: Path = Path("data/index.sqlite3")
    retrieval: Literal["lexical", "hybrid"] = "lexical"
    generator: Literal["extractive", "transformers"] = "extractive"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str | None = None
    llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    llm_revision: str | None = None
    llm_device: Literal["cpu", "cuda", "mps"] = "cpu"
    llm_context_tokens: int = Field(default=4096, ge=512, le=32768)
    max_new_tokens: int = Field(default=512, ge=32, le=2048)
    api_key: str = Field(default="", repr=False)
    cors_origins: list[str] = []
    chunk_words: int = Field(default=160, ge=20, le=400)
    overlap_words: int = Field(default=30, ge=0)
    max_file_bytes: int = Field(default=10_000_000, ge=1)
    max_document_chars: int = Field(default=1_000_000, ge=100)
    max_chunks: int = Field(default=10_000, ge=1)
    context_chars: int = Field(default=6000, ge=500, le=20_000)
    min_dense_score: float = Field(default=0.35, ge=-1, le=1)
    requests_per_minute: int = Field(default=30, ge=1)
    concurrent_queries: int = Field(default=2, ge=1, le=16)
    generation_seconds: float = Field(default=90, gt=0, le=300)

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.overlap_words >= self.chunk_words:
            raise ValueError("overlap_words must be smaller than chunk_words")
        return self
