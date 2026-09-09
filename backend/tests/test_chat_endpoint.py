import pytest

from app.core.exceptions import LLMProviderError
from app.db.models import Chunk
from app.retrieval import service as retrieval_service
from app.retrieval.vector_store import ScoredChunk


def _scored_chunk(similarity: float, filename: str = "handbook.pdf", page: int = 3) -> ScoredChunk:
    chunk = Chunk(page_number=page, chunk_index=0, content="Employees get 30 days of paid leave.")
    return ScoredChunk(chunk=chunk, filename=filename, similarity=similarity)


def test_chat_returns_answer_with_sources(client, fake_llm, monkeypatch):
    monkeypatch.setattr(retrieval_service, "has_any_documents", lambda db: True)
    monkeypatch.setattr(
        retrieval_service, "search_similar_chunks", lambda db, emb, top_k: [_scored_chunk(0.9)]
    )

    response = client.post("/chat", json={"question": "How many vacation days do I get?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == fake_llm.answer
    assert body["cached"] is False
    assert body["sources"] == [
        {"filename": "handbook.pdf", "page": 3, "excerpt": "Employees get 30 days of paid leave."}
    ]


def test_chat_is_cached_on_repeated_question(client, fake_llm, monkeypatch):
    monkeypatch.setattr(retrieval_service, "has_any_documents", lambda db: True)
    monkeypatch.setattr(
        retrieval_service, "search_similar_chunks", lambda db, emb, top_k: [_scored_chunk(0.9)]
    )

    first = client.post("/chat", json={"question": "How many vacation days do I get?"})
    second = client.post("/chat", json={"question": "  how many VACATION days do I get?  "})

    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    assert fake_llm.calls == 1  # the LLM must not be called twice for the same question


def test_chat_without_documents_returns_409(client, monkeypatch):
    monkeypatch.setattr(retrieval_service, "has_any_documents", lambda db: False)

    response = client.post("/chat", json={"question": "What is the refund policy?"})

    assert response.status_code == 409
    assert response.json()["error"] == "empty_knowledge_base"


def test_chat_with_low_relevance_refuses_to_answer(client, fake_llm, monkeypatch):
    monkeypatch.setattr(retrieval_service, "has_any_documents", lambda db: True)
    monkeypatch.setattr(
        retrieval_service, "search_similar_chunks", lambda db, emb, top_k: [_scored_chunk(0.05)]
    )

    response = client.post("/chat", json={"question": "What is the capital of France?"})

    assert response.status_code == 422
    assert response.json()["error"] == "no_relevant_context"
    assert fake_llm.calls == 0  # never spend LLM tokens on insufficient context


def test_chat_llm_failure_returns_502(client, monkeypatch):
    from tests.conftest import FakeLLMProvider
    from app.llm.providers import get_llm_provider
    from app.main import app

    monkeypatch.setattr(retrieval_service, "has_any_documents", lambda db: True)
    monkeypatch.setattr(
        retrieval_service, "search_similar_chunks", lambda db, emb, top_k: [_scored_chunk(0.9)]
    )
    app.dependency_overrides[get_llm_provider] = lambda: FakeLLMProvider(
        error=LLMProviderError("provider is down")
    )

    response = client.post("/chat", json={"question": "How many vacation days do I get?"})

    assert response.status_code == 502
    assert response.json()["error"] == "llm_provider_error"


def test_chat_rejects_too_short_question(client):
    response = client.post("/chat", json={"question": "hi"})
    assert response.status_code == 422
