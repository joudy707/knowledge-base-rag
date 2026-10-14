
import pytest

from app.retrieval.fusion import reciprocal_rank_fusion


def test_fusion_prefers_consistently_retrieved_chunk():
    semantic_results = [
        {"text": "Chunk A", "metadata": {"source": "a.txt"}},
        {"text": "Chunk B", "metadata": {"source": "b.txt"}},
    ]

    bm25_results = [
        {"text": "Chunk B", "metadata": {"source": "b.txt"}},
        {"text": "Chunk A", "metadata": {"source": "a.txt"}},
    ]

    results = reciprocal_rank_fusion(
        [semantic_results, bm25_results],
        top_k=2,
    )

    assert len(results) == 2
    # Both chunks have the same two ranks, so their scores are equal.
    assert results[0]["rrf_score"] == pytest.approx(
        results[1]["rrf_score"]
    )


def test_chunk_ranked_first_in_both_lists_wins():
    first_list = [
        {"text": "Best", "metadata": {}},
        {"text": "Other", "metadata": {}},
    ]

    second_list = [
        {"text": "Best", "metadata": {}},
        {"text": "Other", "metadata": {}},
    ]

    results = reciprocal_rank_fusion(
        [first_list, second_list],
        top_k=2,
    )

    assert results[0]["text"] == "Best"
    assert results[0]["rrf_score"] > results[1]["rrf_score"]


def test_top_k_is_respected():
    results = reciprocal_rank_fusion(
        [[
            {"text": "A", "metadata": {}},
            {"text": "B", "metadata": {}},
            {"text": "C", "metadata": {}},
        ]],
        top_k=2,
    )

    assert len(results) == 2


def test_empty_result_lists_return_empty_results():
    assert reciprocal_rank_fusion([], top_k=3) == []
    assert reciprocal_rank_fusion([[]], top_k=3) == []


def test_invalid_arguments_raise_errors():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], top_k=0)

    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], rrf_k=0)


def test_duplicate_chunk_in_same_list_is_not_counted_twice():
    results = reciprocal_rank_fusion(
        [[
            {"text": "A", "metadata": {}},
            {"text": "A", "metadata": {}},
        ]],
        top_k=3,
    )

    assert len(results) == 1
    assert results[0]["rrf_score"] == pytest.approx(1 / 61)


def test_result_without_text_is_rejected():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[{"metadata": {}}]])
