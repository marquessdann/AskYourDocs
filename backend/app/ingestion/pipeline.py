from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Chunk, Document
from app.ingestion.chunking import chunk_text
from app.ingestion.pdf_loader import extract_pages
from app.llm.embeddings import EmbeddingProvider


@dataclass(frozen=True)
class IngestResult:
    document: Document
    chunks_created: int


def ingest_pdf(
    db: Session,
    filename: str,
    file_bytes: bytes,
    embedding_provider: EmbeddingProvider,
) -> IngestResult:
    """Extracts text per page, chunks each page with overlap, embeds every chunk in one
    batch call, and persists Document + Chunk rows. Page numbers are preserved per-chunk
    so answers can later cite an exact page, not just a filename."""
    pages = extract_pages(file_bytes)

    chunk_records: list[tuple[int, int, str]] = []  # (page_number, chunk_index, content)
    for page in pages:
        for chunk in chunk_text(page.text, settings.chunk_size_chars, settings.chunk_overlap_chars):
            chunk_records.append((page.page_number, chunk.index, chunk.content))

    contents = [content for _, _, content in chunk_records]
    embeddings = embedding_provider.embed(contents) if contents else []

    document = Document(filename=filename, pages=len(pages))
    db.add(document)
    db.flush()  # assigns document.id without committing yet

    for (page_number, chunk_index, content), embedding in zip(chunk_records, embeddings, strict=True):
        db.add(
            Chunk(
                document_id=document.id,
                page_number=page_number,
                chunk_index=chunk_index,
                content=content,
                embedding=embedding,
            )
        )

    db.commit()
    db.refresh(document)
    return IngestResult(document=document, chunks_created=len(chunk_records))
