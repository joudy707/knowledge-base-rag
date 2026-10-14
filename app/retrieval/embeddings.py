
from sentence_transformers import SentenceTransformer


# Multilingual model suitable for an initial semantic-search project.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_model = None


def get_model() -> SentenceTransformer:
    
    #Load the embedding model once and reuse it.
    
    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def embed_texts(texts: list[str]):
   
   # Convert a list of texts into numerical vectors.
    #Returns a NumPy array with one vector per input text.

    if not texts:
        raise ValueError("texts must contain at least one text")

    if any(not text.strip() for text in texts):
        raise ValueError("texts cannot contain empty strings")

    model = get_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings


def embed_query(query: str):
    
    #Convert  user query into an embedding vector(a single query).
    
    if not query.strip():
        raise ValueError("query cannot be empty")

    return embed_texts([query])[0]
