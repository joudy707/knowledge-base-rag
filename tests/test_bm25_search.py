
import pytest

from app.retrieval.bm25_search import (
    build_bm25_index,
    search_bm25,
)


@pytest.fixture
def sample_chunks():
    return [
        {
            "text": "FastAPI is a Python framework for building APIs.",
            "metadata": {"source": "backend.txt"},
        },
        {
            "text": "Redis is an in-memory data store used for caching.",
            "metadata": {"source": "redis.txt"},
        },
        {
            "text": "JWT tokens can be used for authentication.",
            "metadata": {"source": "auth.txt"},
        },
    ]


def test_build_bm25_index(sample_chunks):
    index = build_bm25_index(sample_chunks)

    assert index["bm25"] is not None
    assert len(index["chunks"]) == 3


def test_search_returns_relevant_chunk(sample_chunks):
    index = build_bm25_index(sample_chunks)

    results = search_bm25(
        "Python framework APIs",
        index,
        top_k=3,
    )

    assert len(results) == 3
    assert "FastAPI" in results[0]["text"]
    assert results[0]["metadata"]["source"] == "backend.txt"


def test_search_respects_top_k(sample_chunks):
    index = build_bm25_index(sample_chunks)

    results = search_bm25(
        "Python",
        index,
        top_k=1,
    )

    assert len(results) == 1


def test_empty_query_raises_error(sample_chunks):
    index = build_bm25_index(sample_chunks)

    with pytest.raises(ValueError):
        search_bm25("", index)


def test_empty_chunks_raise_error():
    with pytest.raises(ValueError):
        build_bm25_index([])
