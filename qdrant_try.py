import os, json, requests, numpy as np

QDRANT = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
COL = os.getenv("QDRANT_COLLECTION", "chunks")
DIM = 384


def ensure_collection():
    r = requests.put(
        f"{QDRANT}/collections/{COL}",
        json={"vectors": {"size": DIM, "distance": "Cosine"}},
    )
    if r.status_code not in (200, 201, 409):
        raise SystemExit(f"ensure_collection failed: {r.status_code} {r.text}")


def insert_points():
    v1 = np.random.randn(DIM).astype("float32").tolist()
    v2 = np.random.randn(DIM).astype("float32").tolist()
    points = [
        {
            "id": 1,
            "vector": v1,
            "payload": {"doc_id": "demo", "chunk_index": 0, "text": "hello world"},
        },
        {
            "id": 2,
            "vector": v2,
            "payload": {"doc_id": "demo", "chunk_index": 1, "text": "second chunk"},
        },
    ]
    r = requests.put(f"{QDRANT}/collections/{COL}/points", json={"points": points})
    r.raise_for_status()
    print("upsert ok:", r.json())


def search():
    qvec = np.random.randn(DIM).astype("float32").tolist()
    r = requests.post(
        f"{QDRANT}/collections/{COL}/points/search", json={"vector": qvec, "limit": 5}
    )
    r.raise_for_status()
    print("search result:\n", json.dumps(r.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ensure_collection()
    insert_points()
    search()
