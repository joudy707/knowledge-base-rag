
import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    #Convert text into lowercase word tokens.
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def build_bm25_index(chunks: list[dict]) -> dict:
    #Build a BM25 index from document chunks.
    if not chunks:
        raise ValueError("chunks cannot be empty")

    tokenized_chunks = []

    for chunk in chunks:
        text = chunk.get("text")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Every chunk must have non-empty text"
            )

        tokens = tokenize(text)

        if not tokens:
            raise ValueError(
                "Every chunk must contain searchable tokens"
            )

        tokenized_chunks.append(tokens)

    bm25 = BM25Okapi(tokenized_chunks)

    return {
        "chunks": chunks,
        "bm25": bm25,
    }


def search_bm25(query: str,index: dict,top_k: int = 3,) -> list[dict]:
    #Return the top matching chunks using BM25.
    if not query.strip():
        raise ValueError("query cannot be empty")

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    chunks = index["chunks"]
    bm25 = index["bm25"]

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results = []

    for i in ranked_indices:
        results.append({
            "text": chunks[i]["text"],
            "metadata": chunks[i].get("metadata", {}),
            "score": float(scores[i]),
        })

    return results
