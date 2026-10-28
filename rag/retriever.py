
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# --------------------------------------------------
# Load embedding model from local cache
# --------------------------------------------------

model = SentenceTransformer(
    EMBEDDING_MODEL,
    # local_files_only=True
)

# --------------------------------------------------
# Connect to existing ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

collection = client.get_collection(
    "innerglow_knowledge"
)

# --------------------------------------------------
# Retrieve relevant knowledge
# --------------------------------------------------

def retrieve_knowledge(query: str, top_k: int = 3):

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
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
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
