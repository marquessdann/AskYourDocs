from functools import lru_cache

from sqlalchemy.orm import Session

from app.cache.query_cache import TTLCache
from app.config import settings
from app.core.exceptions import EmptyKnowledgeBaseError, NoRelevantContextError
from app.llm.embeddings import EmbeddingProvider
from app.llm.prompts import SYSTEM_PROMPT, build_user_prompt, format_context_block
from app.llm.providers import LLMProvider
from app.retrieval.vector_store import ScoredChunk, has_any_documents, search_similar_chunks
from app.schemas import ChatResponse, SourceRef


@lru_cache
def get_query_cache() -> TTLCache:
    return TTLCache(max_size=settings.cache_max_size, ttl_seconds=settings.cache_ttl_seconds)


def _to_sources(scored_chunks: list[ScoredChunk]) -> list[SourceRef]:
    seen: set[tuple[str, int]] = set()
    sources: list[SourceRef] = []
    for scored in scored_chunks:
        key = (scored.filename, scored.chunk.page_number)
        if key in seen:
            continue
        seen.add(key)
        excerpt = scored.chunk.content[:240] + ("…" if len(scored.chunk.content) > 240 else "")
        sources.append(SourceRef(filename=scored.filename, page=scored.chunk.page_number, excerpt=excerpt))
    return sources


def answer_question(
    db: Session,
    question: str,
    embedding_provider: EmbeddingProvider,
    llm_provider: LLMProvider,
    cache: TTLCache,
    top_k: int | None = None,
) -> ChatResponse:
    """Full RAG flow: cache lookup -> retrieval -> relevance gate -> prompt -> LLM -> sources.

    The relevance gate is what keeps this from hallucinating: if the best matching chunk
    is below `min_relevance_score`, we raise instead of asking the LLM to answer from
    context it doesn't really have.
    """
    cached = cache.get(question)
    if cached is not None:
        return ChatResponse(answer=cached.answer, sources=cached.sources, cached=True)

    if not has_any_documents(db):
        raise EmptyKnowledgeBaseError("No documents have been uploaded yet.")

    [query_embedding] = embedding_provider.embed([question])
    scored_chunks = search_similar_chunks(db, query_embedding, top_k or settings.retrieval_top_k)

    best_score = scored_chunks[0].similarity if scored_chunks else -1.0
    if best_score < settings.min_relevance_score:
        raise NoRelevantContextError(best_score=best_score, threshold=settings.min_relevance_score)

    context_blocks = [
        format_context_block(scored.filename, scored.chunk.page_number, scored.chunk.content)
        for scored in scored_chunks
    ]
    user_prompt = build_user_prompt(question, context_blocks)
    answer_text = llm_provider.generate(SYSTEM_PROMPT, user_prompt)

    response = ChatResponse(answer=answer_text, sources=_to_sources(scored_chunks), cached=False)
    cache.set(question, response)
    return response
