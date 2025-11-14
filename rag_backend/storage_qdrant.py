# rag_backend/storage_qdrant.py
import os
import requests
from typing import List, Dict
from uuid import uuid4  # to generate valid Qdrant id

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333").rstrip(
    "/"
)  # default Qdrant url
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "chunks")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
# Normalize distance to Qdrant expected enum strings
_dist_raw = (os.getenv("QDRANT_DISTANCE", "cosine") or "").strip().lower()
if _dist_raw in ("cosine", "cos"):
    DISTANCE = "Cosine"
elif _dist_raw in ("dot", "dotproduct", "dot_product"):
    DISTANCE = "Dot"
elif _dist_raw in ("euclid", "l2", "euclidean"):
    DISTANCE = "Euclid"
else:
    DISTANCE = "Cosine"


class QdrantStorage:
    def __init__(self):
        # optional: clear Qdrant collection on startup if env var is set
        reset_on_startup = os.getenv("RESET_ON_STARTUP", "false").lower() in (
            "1",
            "true",
            "yes",
        )
        if reset_on_startup:
            try:
                requests.delete(
                    f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}", timeout=10
                )
                print(
                    f"[INIT] Qdrant collection '{QDRANT_COLLECTION}' cleared on startup."
                )
            except Exception as e:
                print(f"[INIT] Qdrant clear failed: {e}")
        self._ensure_collection()

    # Create-if-not-exists (idempotent). Qdrant returns 409 if already exists.
    def _ensure_collection(self) -> None:
        payload = {"vectors": {"size": EMBED_DIM, "distance": DISTANCE}}
        r = requests.put(
            f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}",
            json=payload,
            timeout=15,
        )
        if r.status_code not in (200, 409):
            r.raise_for_status()
        # sanity check: ensure collection schema exists
        rc = requests.get(f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}", timeout=10)
        rc.raise_for_status()

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

    def _wait_deleted(self, timeout_sec: int = 20):
        """Poll until the collection is actually gone (DELETE may be async 202)."""
        import time

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            gr = requests.get(
                f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}", timeout=5
            )
            if gr.status_code == 404:
                return True
            time.sleep(0.3)
        return False

    def clear(self) -> Dict:
        """
        Clear all points in the existing collection without touching its schema.
        This avoids subtle DROP + CREATE issues and keeps vectors_config unchanged.
        """
        body = {
            "filter": {
                # empty "must" matches all points in the collection
                "must": []
            }
        }
        r = requests.post(
            f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/delete",
            json=body,
            timeout=30,
        )
        r.raise_for_status()

        return {"ok": True, "collection": QDRANT_COLLECTION, "reset": True}
