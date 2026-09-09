from abc import ABC, abstractmethod
from functools import lru_cache

from app.config import Settings, settings
from app.core.exceptions import LLMProviderError


class EmbeddingProvider(ABC):
    dimension: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Returns one embedding vector per input text, same order."""


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, cfg: Settings):
        if not cfg.openai_api_key:
            raise LLMProviderError("OPENAI_API_KEY is required for embedding_provider=openai")
        from openai import OpenAI

        self._client = OpenAI(api_key=cfg.openai_api_key)
        self._model = cfg.openai_embedding_model
        self.dimension = cfg.embedding_dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(model=self._model, input=texts)
        except Exception as exc:  # noqa: BLE001 - surfaced as a domain error for the API layer
            raise LLMProviderError(f"OpenAI embeddings request failed: {exc}") from exc
        return [item.embedding for item in response.data]


class GoogleEmbeddingProvider(EmbeddingProvider):
    """Uses the Gemini API's embeddings endpoint: free tier, no credit card required."""

    def __init__(self, cfg: Settings):
        if not cfg.google_api_key:
            raise LLMProviderError("GOOGLE_API_KEY is required for embedding_provider=google")
        from google import genai

        self._client = genai.Client(api_key=cfg.google_api_key)
        self._model = cfg.google_embedding_model
        self.dimension = cfg.embedding_dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        from google.genai.types import EmbedContentConfig

        try:
            response = self._client.models.embed_content(
                model=self._model,
                contents=texts,
                config=EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self.dimension,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(f"Gemini embeddings request failed: {exc}") from exc
        return [embedding.values for embedding in response.embeddings]


class LocalEmbeddingProvider(EmbeddingProvider):
    """Runs a small sentence-transformers model locally: free, offline, no API key."""

    def __init__(self, cfg: Settings):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(cfg.local_embedding_model)
        self.dimension = cfg.embedding_dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self._model.encode(texts, normalize_embeddings=True)
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(f"Local embedding model failed: {exc}") from exc
        return [vector.tolist() for vector in vectors]


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(settings)
    if settings.embedding_provider == "google":
        return GoogleEmbeddingProvider(settings)
    return LocalEmbeddingProvider(settings)
