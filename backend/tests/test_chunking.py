import pytest

from app.ingestion.chunking import chunk_text


def test_short_text_returns_single_chunk():
    chunks = chunk_text("Hello world.", chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].content == "Hello world."
    assert chunks[0].index == 0


def test_empty_text_returns_no_chunks():
    assert chunk_text("   ", chunk_size=100, overlap=10) == []
    assert chunk_text("", chunk_size=100, overlap=10) == []


def test_long_text_is_split_into_multiple_chunks():
    text = "word " * 500  # 2500 chars
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.content) <= 200 + 20  # boundary snapping can extend slightly


def test_chunks_are_sequentially_indexed():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_overlap():
    text = "abcdefghij " * 100
    chunks = chunk_text(text, chunk_size=100, overlap=30)
    first_tail = chunks[0].content[-15:]
    assert first_tail in chunks[1].content


def test_no_content_is_lost_without_overlap():
    text = "one two three four five six seven eight nine ten"
    chunks = chunk_text(text, chunk_size=15, overlap=0)
    rebuilt = " ".join(c.content for c in chunks)
    assert set(text.split()) == set(rebuilt.split())


@pytest.mark.parametrize("overlap", [-1, 50])
def test_invalid_overlap_raises(overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=50, overlap=overlap)


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=0, overlap=0)
