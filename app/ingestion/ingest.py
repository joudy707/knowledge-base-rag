
from pathlib import Path

from app.ingestion.parser import load_document
from app.ingestion.chunker import split_text


def ingest_document(file_path: str) -> list[dict]:
    #Load a document and return chunks with metadata

    path = Path(file_path)
    text = load_document(file_path)

    chunks = split_text(text,chunk_size=200,overlap=30,)

    results = []

    for index, chunk in enumerate(chunks):
        results.append(
            {
                "chunk_id": f"{path.stem}_{index}",
                "text": chunk,
                "metadata": {
                    "source": path.name,
                    "chunk_index": index,
                },
            }
        )

    return results
