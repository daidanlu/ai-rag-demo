# rag_backend/storage_factory.py
import os
from .storage_qdrant import QdrantStorage
from .storage_memory import MemoryStorage


#   Factory: return a storage instance based on RAG_STORAGE env.
#   RAG_STORAGE = 'qdrant' -> QdrantStorage
#                anything else -> MemoryStorage


def get_storage():
    mode = os.environ.get("RAG_STORAGE", "memory").lower()
    if mode == "qdrant":
        return QdrantStorage()
    return MemoryStorage()


# Unified clear/reset entry. Prefer .clear(); fallback to .reset().
def clear_storage():
    s = get_storage()
    if hasattr(s, "clear"):
        return s.clear()
    if hasattr(s, "reset"):
        return s.reset()
    return {"ok": False, "message": "clear not supported"}
