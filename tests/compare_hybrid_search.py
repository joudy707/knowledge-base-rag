
from app.retrieval.bm25_search import (
    build_bm25_index,
    search_bm25,
)
from app.retrieval.vector_search import (
    build_vector_index,
    search_chunks,
)
from app.retrieval.fusion import reciprocal_rank_fusion


chunks = [
    {
        "text": "FastAPI is a Python framework for building APIs.",
        "metadata": {
            "source": "backend.txt",
            "chunk_index": 0,
        },
    },
    {
        "text": "Redis is an in-memory data store used for caching.",
        "metadata": {
            "source": "redis.txt",
            "chunk_index": 0,
        },
    },
    {
        "text": "JWT tokens can be used for authentication.",
        "metadata": {
            "source": "auth.txt",
            "chunk_index": 0,
        },
    },
    {
        "text": "FastAPI supports building backend web services.",
        "metadata": {
            "source": "backend.txt",
            "chunk_index": 1,
        },
    },
]

query = "How can I build a Python API?"

# Build each index once.
bm25_index = build_bm25_index(chunks)
vector_index = build_vector_index(chunks)

# Retrieve candidates from both methods.
bm25_results = search_bm25(query, bm25_index, top_k=4)
semantic_results = search_chunks(query, vector_index, top_k=4)

# Merge their rankings.
hybrid_results = reciprocal_rank_fusion(
    [semantic_results, bm25_results],
    top_k=3,
)

print("\n--- Hybrid Search Results ---")

for result in hybrid_results:
    print("RRF score:", round(result["rrf_score"], 5))
    print("Ranks:", result["ranks"])
    print("Source:", result["metadata"].get("source"))
    print("Text:", result["text"])
    print("-" * 40)
