import pytest
from fastapi.testclient import TestClient

from app.api.deps import enforce_rate_limit
from app.cache.query_cache import TTLCache
from app.db.session import get_db
from app.llm.embeddings import get_embedding_provider
from app.llm.providers import get_llm_provider
from app.main import app
from app.retrieval.service import get_query_cache


class FakeEmbeddingProvider:
    dimension = 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dimension for _ in texts]


class FakeLLMProvider:
    def __init__(self, answer: str = "This is the answer.", error: Exception | None = None):
        self.answer = answer
        self.error = error
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        if self.error:
            raise self.error
        return self.answer


@pytest.fixture
def fake_llm() -> FakeLLMProvider:
    return FakeLLMProvider()


@pytest.fixture
def client(fake_llm: FakeLLMProvider):
    app.dependency_overrides[get_db] = lambda: iter([None])
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_llm_provider] = lambda: fake_llm
    cache = TTLCache(max_size=16, ttl_seconds=3600)
    app.dependency_overrides[get_query_cache] = lambda: cache
    app.dependency_overrides[enforce_rate_limit] = lambda: None

    # Not used as a context manager: that would run the startup event (init_db),
    # which tries to open a real Postgres connection we don't have in tests.
    yield TestClient(app)

    app.dependency_overrides.clear()
