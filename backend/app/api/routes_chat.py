from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import enforce_rate_limit
from app.cache.query_cache import TTLCache
from app.db.session import get_db
from app.llm.embeddings import EmbeddingProvider, get_embedding_provider
from app.llm.providers import LLMProvider, get_llm_provider
from app.retrieval.service import answer_question, get_query_cache
from app.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse, dependencies=[Depends(enforce_rate_limit)])
def chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    llm_provider: LLMProvider = Depends(get_llm_provider),
    cache: TTLCache = Depends(get_query_cache),
) -> ChatResponse:
    return answer_question(
        db=db,
        question=payload.question,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        cache=cache,
        top_k=payload.top_k,
    )
