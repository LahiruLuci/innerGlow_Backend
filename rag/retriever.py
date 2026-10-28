
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "chroma_db"
PDF_FOLDER = BASE_DIR / "knowledge" / "pdfs"

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

COLLECTION_NAME = "innerglow_knowledge"

model = SentenceTransformer(EMBEDDING_MODEL)

client = chromadb.PersistentClient(path=str(DB_PATH))


def load_collection():
    """Load the knowledge collection, building it if missing."""

    try:
        existing = client.get_collection(COLLECTION_NAME)

        if existing.count() > 0:
            return existing

    except NotFoundError:
        pass

    if not list(PDF_FOLDER.glob("*.pdf")):
        raise RuntimeError(
            "RAG collection is missing and no source PDFs "
            "were found in knowledge/pdfs. "
            "Make sure the PDFs are included in GitHub."
        )

    print("RAG collection missing. Building PDF index...")

    from rag.build_index import build_index

    build_index()

    return client.get_collection(COLLECTION_NAME)


collection = load_collection()


def retrieve_knowledge(query: str, top_k: int = 3):
    if not query.strip() or top_k <= 0:
        return []

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    collection_size = collection.count()

    if collection_size == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection_size),
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []

    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        metadata = metadata or {}

        retrieved.append({
            "text": text,
            "topic": metadata.get("topic", "General"),
            "source": metadata.get("source", "Unknown"),
            "distance": distance
        })

    return retrieved
