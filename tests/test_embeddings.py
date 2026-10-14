
from app.retrieval.embeddings import embed_texts, embed_query


def test_embeddings():
    texts = [
        "FastAPI is a Python web framework.",
        "Python is used for backend development.",
        "The weather is sunny today.",
    ]

    embeddings = embed_texts(texts)
    query_vector = embed_query("What is FastAPI?")

    assert embeddings.shape == (3, 384)
    assert query_vector.shape == (384,)

    print("Embedding test passed!")
