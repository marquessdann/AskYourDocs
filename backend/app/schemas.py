from datetime import datetime

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: int
    filename: str
    pages: int
    chunks_created: int
    uploaded_at: datetime


class DocumentSummary(BaseModel):
    id: int
    filename: str
    pages: int
    uploaded_at: datetime


class SourceRef(BaseModel):
    filename: str
    page: int
    excerpt: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    cached: bool = False


class ErrorResponse(BaseModel):
    error: str
    detail: str
