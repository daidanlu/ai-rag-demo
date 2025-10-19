# rag_backend/storage_qdrant.py
import os
import requests
from typing import List, Dict
from uuid import uuid4  # to generate valid Qdrant id

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")  # default Qdrant url
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "chunks")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
DISTANCE = os.getenv("QDRANT_DISTANCE", "Cosine")


class QdrantStorage:
    def __init__(self):
        self._ensure_collection()

    # Create-if-not-exists. Qdrant returns 409 if the collection already exists.Treat 200/409 as OK; only raise for other codes.
    def _ensure_collection(self) -> None:
        r = requests.put(
            f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}",
            json={"vectors": {"size": EMBED_DIM, "distance": DISTANCE}},
            timeout=10,
        )
        if r.status_code not in (200, 409):
            r.raise_for_status()

    # to write into Qdrant
    def upsert(self, vectors, payloads):
        points = []
        for vec, pld in zip(vectors, payloads):
            pid = str(uuid4())  # generate valid id for Qdrant
            points.append(
                {"id": pid, "vector": vec, "payload": pld}
            )  # create a dict for (vec, pld) pairs and add to points list

        r = requests.put(
            f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points",
            json={"points": points},
            timeout=30,
        )
        if r.status_code >= 400:
            print(">>> Qdrant upsert error", r.status_code, r.text)
        r.raise_for_status()
        return r.json()

    def search(self, query_vec: List[float], k: int = 5, filters: Dict = None):
        body = {
            "vector": query_vec,
            "limit": k,
            "with_payload": True,
            "with_vectors": False,
        }
        if filters:
            body["filter"] = filters

        r = requests.post(
            f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search",
            json=body,
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("result", [])

    def clear(self) -> Dict:
        # drop collection
        r_del = requests.delete(
            f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}",
            timeout=30,
        )
        # not exist
        if r_del.status_code not in (200, 202, 404):
            r_del.raise_for_status()

        # rebuild collection
        self._ensure_collection()
        return {"ok": True, "collection": QDRANT_COLLECTION, "reset": True}
