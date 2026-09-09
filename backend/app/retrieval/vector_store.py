from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    filename: str
    similarity: float  # cosine similarity in [-1, 1], higher = more relevant


def search_similar_chunks(db: Session, query_embedding: list[float], top_k: int) -> list[ScoredChunk]:
    """Finds the top_k chunks whose embedding is closest to the query, using pgvector's
    cosine distance operator. cosine_distance() returns 1 - cosine_similarity, so we
    convert back to a similarity score for readability at the call site."""
    distance = Chunk.embedding.cosine_distance(query_embedding)
    stmt = (
        select(Chunk, Document.filename, distance.label("distance"))
        .join(Document, Document.id == Chunk.document_id)
        .order_by(distance)
        .limit(top_k)
    )
    rows = db.execute(stmt).all()
    return [
        ScoredChunk(chunk=chunk, filename=filename, similarity=1 - dist)
        for chunk, filename, dist in rows
    ]


def has_any_documents(db: Session) -> bool:
    return db.execute(select(Document.id).limit(1)).first() is not None
