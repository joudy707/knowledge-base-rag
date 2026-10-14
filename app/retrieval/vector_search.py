
import numpy as np

from app.retrieval.embeddings import embed_texts, embed_query


def build_vector_index(chunks: list[dict]) -> list[dict]:
    
    # Generate and attach one embedding to each chunk.
    # Each chunk must contain:
    # - text: str
    # - metadata: dict

    if not chunks:
        raise ValueError("chunks cannot be empty")

    texts = []

    for chunk in chunks:
        text = chunk.get("text")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Every chunk must have non-empty text")

        texts.append(text)

    embeddings = embed_texts(texts)

    indexed_chunks = []

    for chunk, embedding in zip(chunks, embeddings):
        indexed_chunks.append({
            "text": chunk["text"],
            "metadata": chunk.get("metadata", {}),
            "embedding": embedding,
        })

    return indexed_chunks


def search_chunks(query: str,indexed_chunks: list[dict],top_k: int = 3,) -> list[dict]:

   # Return the top_k chunks with the highest cosine similarity.
    
    if not query.strip():
        raise ValueError("query cannot be empty")

    if not indexed_chunks:
        return []

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    query_embedding = embed_query(query)

    chunk_embeddings = np.vstack([
        chunk["embedding"]
        for chunk in indexed_chunks
    ])

    # Both query and chunk vectors are normalized,
    # so their dot product equals cosine similarity.
    scores = chunk_embeddings @ query_embedding

    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in ranked_indices:
        chunk = indexed_chunks[int(index)]

        results.append({
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "score": float(scores[index]),
        })

    return results
