
from app.ingestion.parser import load_document

text = load_document("data/sample.txt")

print("Extracted text:")
print(text)

print("\nTotal characters:", len(text))
