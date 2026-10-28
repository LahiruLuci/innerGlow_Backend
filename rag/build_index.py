
from pathlib import Path
import hashlib

import chromadb
from chromadb.errors import NotFoundError
import fitz
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
PDF_FOLDER = BASE_DIR / "knowledge" / "pdfs"
DB_PATH = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

COLLECTION_NAME = "innerglow_knowledge"


def split_text(text, chunk_size=800, overlap=150):
    """Split text into overlapping character chunks."""

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def build_index():

    pdf_files = sorted(
        PDF_FOLDER.glob("*.pdf")
    )

    if not pdf_files:
        print("No PDF files found!")
        return

    print(f"Found {len(pdf_files)} PDF files.")

    model = SentenceTransformer(EMBEDDING_MODEL)

    documents = []
    metadatas = []
    ids = []

    for pdf_path in pdf_files:

        print(f"Processing: {pdf_path.name}")

        with fitz.open(pdf_path) as pdf:

            for page_number, page in enumerate(pdf, start=1):

                text = page.get_text("text")

                chunks = split_text(text)

                for chunk_number, chunk in enumerate(chunks):

                    unique_key = (
                        f"{pdf_path.name}:"
                        f"{page_number}:"
                        f"{chunk_number}"
                    )

                    chunk_id = hashlib.sha256(
                        unique_key.encode("utf-8")
                    ).hexdigest()

                    documents.append(chunk)

                    metadatas.append({
                        "source": pdf_path.name,
                        "page": page_number,
                        "topic": pdf_path.stem
                    })

                    ids.append(chunk_id)

    if not documents:
        print("No readable text found in the PDFs.")
        return

    print(f"Created {len(documents)} text chunks.")

    print("Generating embeddings...")

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True
    ).tolist()

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    # Rebuild the collection so old JSON entries
    # are not mixed with the new PDF chunks.
    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )
    except (ValueError, NotFoundError):
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 100

    for start in range(0, len(documents), batch_size):

        end = start + batch_size

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end]
        )

    print("\nPDF knowledge base created successfully!")
    print(f"Total chunks: {collection.count()}")


if __name__ == "__main__":
    build_index()
