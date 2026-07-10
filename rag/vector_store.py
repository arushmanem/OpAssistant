"""Manages the ChromaDB collection: setup, storage, and retrieval."""
import chromadb
from config import CHROMA_PATH, COLLECTION_NAME


chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection(reset: bool = False):
    """Get the collection, optionally clearing it first."""
    if reset:
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def store_chunks(collection, chunks: list[str], embeddings: list[list[float]]):
    """Add chunks + their embeddings to the collection."""
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks,
    )


def retrieve(collection, question_vector: list[float], n_results: int = 3) -> list[str]:
    """Return the top-N most similar chunks to a query vector."""
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=n_results,
    )
    return results["documents"][0]