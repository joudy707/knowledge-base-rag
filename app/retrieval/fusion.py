
def _result_key(result: dict) -> tuple:
    #Identify a chunk across different retrieval result lists.
    text = result.get("text")

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Every result must have non-empty text")

    metadata = result.get("metadata") or {}

    if not isinstance(metadata, dict):
        raise ValueError("Result metadata must be a dictionary")

    source = str(metadata.get("source", ""))
    chunk_id = str(
        metadata.get("chunk_index", metadata.get("chunk_id", ""))
    )

    return text.strip(), source, chunk_id


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    top_k: int = 3,
    rrf_k: int = 60,
) -> list[dict]:
    """Merge ranked retrieval lists using Reciprocal Rank Fusion."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")

    fused = {}

    for results in result_lists:
        seen_in_list = set()

        for rank, result in enumerate(results, start=1):
            key = _result_key(result)

            # Avoid counting the same chunk twice in one list.
            if key in seen_in_list:
                continue

            seen_in_list.add(key)

            if key not in fused:
                fused[key] = {
                    "text": result["text"],
                    "metadata": result.get("metadata", {}),
                    "rrf_score": 0.0,
                    "ranks": [],
                }

            fused[key]["rrf_score"] += 1.0 / (rrf_k + rank)
            fused[key]["ranks"].append(rank)

    ranked_results = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return ranked_results[:top_k]
