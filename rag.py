# rag.py (RAGService Class using NumPy/Sklearn)
from __future__ import annotations
import glob, json, os, re, sys, uuid
from typing import List, Tuple, Dict, Any
import numpy as np
from pypdf import PdfReader
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

# Configuration (remains global for now)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
DB_DIR = "db"
TEXT_PATH = os.path.join(DB_DIR, "chunks.json")
META_PATH = os.path.join(DB_DIR, "metas.json")
EMB_PATH = os.path.join(DB_DIR, "embeddings.npy")
DEFAULT_GPT4ALL_MODEL = "Llama-3.2-1B-Instruct-Q4_0.gguf"


# Utility Functions (Chunking/Loading PDFs)
def load_pdfs(paths: List[str]) -> List[Tuple[str, str]]:
    docs = []
    for p in paths:
        try:
            reader = PdfReader(p)
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            text = re.sub(r"\s+\n", "\n", text).strip()
            docs.append((os.path.basename(p), text))
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}", file=sys.stderr)
    return docs


def _simple_sentence_split(text: str) -> List[str]:
    return [
        s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", text) if s and s.strip()
    ]


def chunk_text(text: str, max_words: int = 180) -> List[str]:
    sents = _simple_sentence_split(text)
    chunks, buf, count = [], [], 0
    for s in sents:
        w = len(s.split())
        if count + w > max_words and buf:
            chunks.append(" ".join(buf))
            buf, count = [s], w
        else:
            buf.append(s)
            count += w
    if buf:
        chunks.append(" ".join(buf))
    if not chunks and text:
        words = text.split()
        for i in range(0, len(words), max_words):
            chunks.append(" ".join(words[i : i + max_words]))
    return chunks


# RAGService Class: Encapsulates all Core Logic
class RAGService:
    def __init__(self):
        # Service initialization logic (can load models/DBs here if needed)
        pass

    def _ensure_db_dir(self):
        os.makedirs(DB_DIR, exist_ok=True)

    def _load_embedding_model(self):
        return SentenceTransformer(EMBED_MODEL_NAME)

    def _save_index(
        self, embs: np.ndarray, metas: List[Dict[str, Any]], chunks: List[str]
    ):
        # core method to persist RAG indexing in local docs
        self._ensure_db_dir()
        np.save(EMB_PATH, embs)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(metas, f, ensure_ascii=False, indent=2)
        with open(TEXT_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

    def _load_index(self):
        if not (
            os.path.exists(EMB_PATH)
            and os.path.exists(META_PATH)
            and os.path.exists(TEXT_PATH)
        ):
            return None, None, None

        try:
            embs = np.load(EMB_PATH)
            with open(META_PATH, "r", encoding="utf-8") as f:
                metas = json.load(f)
            with open(TEXT_PATH, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            return embs, metas, chunks
        except Exception as e:
            print(f"[ERROR] Failed to load RAG index files: {e}", file=sys.stderr)
            return None, None, None

    def _fit_nn(self, embs: np.ndarray) -> NearestNeighbors:
        # use NN of scikit-learn to train vector indexing model
        nn = NearestNeighbors(
            n_neighbors=min(embs.shape[0], 50), metric="cosine"
        )  # use cosine similarity to search
        nn.fit(embs)
        return nn

    def ingest_files(self, patterns: List[str]) -> int:
        paths = []
        for pat in patterns:
            paths.extend(glob.glob(pat))
        paths = [p for p in sorted(set(paths)) if p.lower().endswith(".pdf")]
        if not paths:
            print("[INFO] No PDFs found.")
            return 0

        docs = load_pdfs(paths)
        model = self._load_embedding_model()

        all_chunks, metas = [], []
        for doc_id, text in docs:
            chunks = chunk_text(text, max_words=180)
            for i, ch in enumerate(chunks):
                all_chunks.append(ch)
                metas.append(
                    {
                        "doc_id": doc_id,
                        "chunk_idx": i,
                        "id": f"{doc_id}:{i}:{uuid.uuid4().hex[:8]}",
                    }
                )

        if not all_chunks:
            print("[WARN] No text chunks extracted.")
            return 0

        embs = model.encode(
            all_chunks,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # to ensure the accuracy of cosine similarity search
        )
        self._save_index(embs, metas, all_chunks)
        print(f"[OK] Ingested {len(docs)} docs, {len(all_chunks)} chunks.")
        return len(all_chunks)

    def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        embs, metas, chunks = self._load_index()
        if embs is None:
            print("[INFO] No index found. Please run ingest first.")
            return []

        # Ensure embs is 2D and non-empty for NearestNeighbors
        if embs.ndim == 1:
            embs = embs.reshape(-1, 1)

        nn = self._fit_nn(embs)
        q_emb = self._load_embedding_model().encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )

        # Handle case where k is greater than available chunks
        n_neighbors = min(k, embs.shape[0])
        if n_neighbors == 0:
            return []

        distances, indices = nn.kneighbors(q_emb, n_neighbors=n_neighbors)
        hits = []
        for idx, dist in zip(indices[0], distances[0]):
            meta = metas[idx]
            hits.append(
                {
                    "id": meta.get("id", f"{meta['doc_id']}:{meta['chunk_idx']}"),
                    "text": chunks[idx],
                    "meta": {"doc_id": meta["doc_id"], "chunk_idx": meta["chunk_idx"]},
                    "distance": float(dist),
                }
            )
        return hits

    # Static LLM-related functions kept outside the class for simplicity
    @staticmethod
    def build_prompt(query: str, hits: List[Dict[str, Any]]) -> str:
        context = "\n\n".join(f"[{i+1}] {h['text']}" for i, h in enumerate(hits))
        return (
            "You are a helpful assistant. Answer ONLY using the provided context.\n"
            "If the answer is not present, reply: 'I don't know based on the given context.'\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )

    @staticmethod
    def call_llamacpp(prompt: str, model_path: str, max_tokens: int = 256) -> str:
        try:
            from llama_cpp import Llama
        except Exception as e:
            return f"[ERROR] llama-cpp-python not installed: {e}"

        # Helper function to resolve model path remains, using global DEFAULT_GPT4ALL_MODEL
        def _resolve_model_path(model_name):
            if os.path.isfile(model_name):
                return model_name

            # Simple search paths
            paths = [os.getcwd()]
            for path in paths:
                candidate = os.path.join(path, model_name)
                if os.path.isfile(candidate):
                    return candidate
            return None

        resolved_model = _resolve_model_path(model_path)
        if not resolved_model:
            return f"[ERROR] Model path does not exist for: {model_path}"

        try:
            m = Llama(
                model_path=resolved_model,
                n_ctx=2048,
                n_gpu_layers=0,  # force to use CPU
                verbose=False,
            )

            output = m.create_completion(
                prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                stop=["\nQuestion:", "Answer:"],
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            return f"[ERROR] Failed to run Llama-CPP ({resolved_model}): {e}"

    def answer(
        self,
        query: str,
        k: int = 4,
        model: str = DEFAULT_GPT4ALL_MODEL,
        max_tokens: int = 256,
        generate: bool = True,
    ) -> Dict[str, Any]:
        hits = self.retrieve(query, k=k)
        if not hits:
            return {"answer": "[No results found]", "hits": []}
        if not generate:
            return {"answer": "\n---\n".join(h["text"] for h in hits), "hits": hits}

        return {
            "answer": self.call_llamacpp(
                self.build_prompt(query, hits), model_path=model, max_tokens=max_tokens
            ),
            "hits": hits,
        }
