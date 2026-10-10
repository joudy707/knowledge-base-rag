
from fastapi import FastAPI

app = FastAPI(
    title="Knowledge Base RAG API",
    description="A learning project for building a RAG backend.",
    version="0.1.0",
)


@app.get("/health", tags=["Health"])
def health_check():
    """Check whether the API is running."""
    return {"status": "ok"}
