from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AskYourDocs"
    environment: str = "development"
    cors_origins: str = "http://localhost:5173,http://localhost:8080,http://127.0.0.1:5500"

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/askyourdocs"

    # --- Embeddings ---
    # "local" uses sentence-transformers (no API key, no cost). "openai" uses text-embedding-3-small.
    embedding_provider: Literal["local", "openai"] = "local"
    embedding_dimension: int = 384  # 384 for the local MiniLM model, 1536 for text-embedding-3-small
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- LLM (answer generation) ---
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_chat_model: str = "gpt-4o-mini"
    anthropic_chat_model: str = "claude-3-5-haiku-latest"

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # --- Ingestion ---
    chunk_size_chars: int = 1000
    chunk_overlap_chars: int = 150
    max_upload_size_mb: int = 20

    # --- Retrieval ---
    retrieval_top_k: int = 5
    # Minimum cosine similarity (1 - cosine distance) a chunk must have to be considered relevant.
    # Below this, we refuse to answer instead of letting the LLM improvise on weak context.
    min_relevance_score: float = 0.25

    # --- Cache & rate limiting (simple in-memory versions, see app/cache/query_cache.py) ---
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 256
    rate_limit_requests: int = 20
    rate_limit_window_seconds: int = 60

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
