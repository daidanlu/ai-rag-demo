# rag_backend/storage_factory.py
import os


def make_storage():
    backend = os.getenv("RAG_STORAGE", "memory")
    if backend == "qdrant":
        from .storage_qdrant import QdrantStorage

        return QdrantStorage()

    return None
