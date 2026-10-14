
from app.retrieval.vector_search import (
    build_vector_index,
    search_chunks,
)


def main():
    # Small knowledge base for our first semantic search.
    chunks = [
        {
            "text": (
                "FastAPI is a Python framework for building APIs. "
                "It supports request validation and asynchronous endpoints."
            ),
            "metadata": {
                "source": "fastapi.txt",
                "chunk_index": 0,
            },
        },
        {
            "text": (
                "Embeddings convert text into numerical vectors "
                "that can represent semantic meaning."
            ),
            "metadata": {
                "source": "rag.txt",
                "chunk_index": 0,
            },
        },
        {
            "text": (
                "Vector search compares a query vector with document "
                "vectors and returns the most similar results."
            ),
            "metadata": {
                "source": "rag.txt",
                "chunk_index": 1,
            },
        },
        {
            "text": (
                "A database stores and organizes information "
                "so that applications can retrieve it."
            ),
            "metadata": {
                "source": "database.txt",
                "chunk_index": 0,
            },
        },
    ]

    # Step 1: Generate and attach embeddings to the chunks.
    index = build_vector_index(chunks)

    # Step 2: Search using a natural-language question.
    query = "What is an API framework?"
    results = search_chunks(
        query=query,
        indexed_chunks=index,
        top_k=3,
    )

    # Step 3: Display the ranked results.
    print(f"\nQuery: {query}\n")

    for rank, result in enumerate(results, start=1):
        print(f"Rank: {rank}")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['metadata']['source']}")
        print(f"Text: {result['text']}")
        print("-" * 50)

    assert len(results) == 3
    assert results[0]["score"] >= results[1]["score"]


if __name__ == "__main__":
    main()
