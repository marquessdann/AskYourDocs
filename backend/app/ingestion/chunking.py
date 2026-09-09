from dataclasses import dataclass

_PARAGRAPH_BREAK = "\n\n"
_SENTENCE_BREAK = ". "


@dataclass(frozen=True)
class TextChunk:
    index: int
    content: str


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[TextChunk]:
    """Splits text into overlapping chunks of roughly `chunk_size` characters.

    Splits are pushed back to the nearest paragraph or sentence boundary when one is
    found within the window, so chunks don't cut words/ideas mid-sentence when avoidable.
    Pure function (no I/O) so it's easy to unit test independently from PDF parsing or DB access.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[TextChunk] = []
    start = 0
    length = len(normalized)

    while start < length:
        end = min(start + chunk_size, length)

        if end < length:
            boundary = normalized.rfind(_PARAGRAPH_BREAK, start, end)
            if boundary == -1:
                boundary = normalized.rfind(_SENTENCE_BREAK, start, end)
                if boundary != -1:
                    boundary += len(_SENTENCE_BREAK) - 1
            if boundary == -1:
                # No paragraph/sentence break in range: fall back to the last word
                # boundary so we never cut a word in half.
                boundary = normalized.rfind(" ", start, end)
            if boundary != -1 and boundary > start:
                end = boundary

        piece = normalized[start:end].strip()
        if piece:
            chunks.append(TextChunk(index=len(chunks), content=piece))

        if end >= length:
            break

        start = max(end - overlap, start + 1)

    return chunks
