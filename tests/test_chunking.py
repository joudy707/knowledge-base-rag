
from app.ingestion.chunker import split_text


def test_empty_text():
    assert split_text("") == []


def test_respects_chunk_size():
    text = "A" * 250

    chunks = split_text(
        text,
        chunk_size=100,
        overlap=20,
    )

    assert all(len(chunk) <= 100 for chunk in chunks)


def test_keeps_short_paragraphs():
    text = "First paragraph.\n\nSecond paragraph."

    chunks = split_text(text, chunk_size=100)

    assert len(chunks) == 1
    assert "First paragraph." in chunks[0]
    assert "Second paragraph." in chunks[0]
