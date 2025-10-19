# rag_backend/storage_memory.py
class MemoryStorage:
    def __init__(self):
        self._vecs = []  # e.g., list[list[float]]
        self._meta = []  # e.g., list[dict]

    def upsert(self, vectors, payloads):
        self._vecs.extend(vectors)
        self._meta.extend(payloads)
        return {"ok": True, "count": len(vectors)}

    def search(self, query_vec, k=5, filters=None):
        # placeholder for demo use: first k items
        hits = []
        for i, meta in enumerate(self._meta[:k]):
            hits.append({"id": i, "payload": meta, "score": 1.0})
        return hits

    def reset(self):
        self._vecs.clear()
        self._meta.clear()
        return {"ok": True, "storage": "memory", "reset": True}

    def clear(self):
        return self.reset()
