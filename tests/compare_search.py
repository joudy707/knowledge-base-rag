
from app.retrieval.bm25_search import (
    build_bm25_index,
    search_bm25,
)
from app.retrieval.vector_search import (
    build_vector_index,
    search_chunks,
)


chunks = [
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

query = "What tool stores data in memory for caching?"

bm25_index = build_bm25_index(chunks)
bm25_results = search_bm25(query, bm25_index, top_k=3)

print("\n--- BM25 Results ---")
for result in bm25_results:
    print(result["score"], result["text"])

vector_index = build_vector_index(chunks)
vector_results = search_chunks(query, vector_index, top_k=3)

print("\n--- Semantic Search Results ---")
for result in vector_results:
    print(result["score"], result["text"])
