
from pathlib import Path
import pymupdf


def load_document(file_path: str) -> str:
    #Extract text from a TXT or PDF document
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8")

    if suffix == ".pdf":
        pages = []

        with pymupdf.open(path) as pdf:
            for page in pdf:
                pages.append(page.get_text())

        return "\n\n".join(pages)

    raise ValueError(f"Unsupported file type: {suffix}")
