from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.core.exceptions import UnsupportedFileError
from app.db.models import Document
from app.db.session import get_db
from app.ingestion.pipeline import ingest_pdf
from app.llm.embeddings import EmbeddingProvider, get_embedding_provider
from app.schemas import DocumentSummary, DocumentUploadResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentUploadResponse, status_code=201)
def upload_document(
    file: UploadFile,
    db: Session = Depends(get_db),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> DocumentUploadResponse:
    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted.")

    file_bytes = file.file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_size_mb}MB limit.")

    try:
        result = ingest_pdf(db, filename=file.filename, file_bytes=file_bytes, embedding_provider=embedding_provider)
    except UnsupportedFileError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return DocumentUploadResponse(
        document_id=result.document.id,
        filename=result.document.filename,
        pages=result.document.pages,
        chunks_created=result.chunks_created,
        uploaded_at=result.document.uploaded_at,
    )


@router.get("", response_model=list[DocumentSummary])
def list_documents(db: Session = Depends(get_db)) -> list[DocumentSummary]:
    documents = db.execute(select(Document).order_by(Document.uploaded_at.desc())).scalars().all()
    return [
        DocumentSummary(id=d.id, filename=d.filename, pages=d.pages, uploaded_at=d.uploaded_at)
        for d in documents
    ]
