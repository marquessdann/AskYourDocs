from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes_chat import router as chat_router
from app.api.routes_documents import router as documents_router
from app.config import settings
from app.core.exceptions import (
    EmptyKnowledgeBaseError,
    LLMProviderError,
    NoRelevantContextError,
    RateLimitExceededError,
    UnsupportedFileError,
)
from app.core.logging import configure_logging
from app.db.session import init_db

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(chat_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _error_response(status_code: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail})


@app.exception_handler(NoRelevantContextError)
def handle_no_relevant_context(request: Request, exc: NoRelevantContextError) -> JSONResponse:
    return _error_response(
        422,
        "no_relevant_context",
        "I couldn't find anything in your documents relevant enough to answer that "
        "confidently, so I'm not going to guess. Try rephrasing, or upload a document "
        "that covers this topic.",
    )


@app.exception_handler(EmptyKnowledgeBaseError)
def handle_empty_kb(request: Request, exc: EmptyKnowledgeBaseError) -> JSONResponse:
    return _error_response(409, "empty_knowledge_base", str(exc))


@app.exception_handler(LLMProviderError)
def handle_llm_error(request: Request, exc: LLMProviderError) -> JSONResponse:
    return _error_response(502, "llm_provider_error", str(exc))


@app.exception_handler(UnsupportedFileError)
def handle_unsupported_file(request: Request, exc: UnsupportedFileError) -> JSONResponse:
    return _error_response(422, "unsupported_file", str(exc))


@app.exception_handler(RateLimitExceededError)
def handle_rate_limit(request: Request, exc: RateLimitExceededError) -> JSONResponse:
    response = _error_response(429, "rate_limit_exceeded", str(exc))
    response.headers["Retry-After"] = str(int(exc.retry_after_seconds))
    return response
