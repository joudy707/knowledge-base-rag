
from app.ingestion.ingest import ingest_document

chunks = ingest_document("data/sample.txt")

print(f"Total chunks: {len(chunks)}")

for chunk in chunks:
    print("\n" + "=" * 40)
    print("ID:", chunk["chunk_id"])
    print("Metadata:", chunk["metadata"])
    print("Text:", chunk["text"])
