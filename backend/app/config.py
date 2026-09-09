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
    # "local" uses sentence-transformers (no API key, but pulls in PyTorch — too heavy for
    # small free-tier hosts). "openai" uses text-embedding-3-small (paid). "google" uses the
    # Gemini API's free tier (no credit card required) — the easiest option for a free deploy.
    embedding_provider: Literal["local", "openai", "google"] = "local"
    embedding_dimension: int = 384  # 384 (local MiniLM), 1536 (text-embedding-3-small), 768 (Gemini)
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"
    google_embedding_model: str = "gemini-embedding-001"

    # --- LLM (answer generation) ---
    # "google" uses the Gemini API's free tier (no credit card) — the easiest free option.
    llm_provider: Literal["openai", "anthropic", "google"] = "openai"
    openai_chat_model: str = "gpt-4o-mini"
    anthropic_chat_model: str = "claude-3-5-haiku-latest"
    google_chat_model: str = "gemini-flash-latest"

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

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
