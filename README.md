# AI RAG Demo (Local LLM Service)

A minimal, secure Retrieval-Augmented Generation (RAG) system built entirely from scratch in Python, focused on running inference and retrieval locally and offline.  
The core logic is exposed via a dedicated Django REST API service.

---

## Core Components

- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Store:** Custom implementation using `NumPy` and `scikit-learn` (`NearestNeighbors`) for memory-based indexing (persisted in `./db`).
- **Generator:** Local LLM inference via `llama-cpp-python` (default model: `Llama-3.2-1B-Instruct-Q4_0.gguf`).

---

## Quickstart & Setup

### A. Backend Service (Django REST API)

This is the core component that handles ingestion, retrieval, and LLM generation.

```bash
# 1. Preparation
# Place your desired GGUF model file (e.g., Llama-3.2-1B-Instruct-Q4_0.gguf) in the project root directory.

# 2. Install Dependencies
pip install -r requirements.txt
pip install -r requirements-web.txt

# 3. Run Service
# Start the Django development server (run this in Terminal 1)
python manage.py runserver
```

---

### B. Command Line Interface (CLI)

Used for direct backend testing or initial indexing.

```bash
# 1. Ingest PDF documents and build the index (index files saved to ./db)
python main.py ingest data/*.pdf

# 2. Ask a question based on your indexed documents
python main.py ask "What is the document about?"
```

---

## Web UI (Streamlit Frontend)

A simple, local web interface for interacting with the Django Backend Service.

```bash
# 1. Start Backend
# Ensure the Django service is running (see section A above)

# 2. Run Frontend
# Launch the Streamlit application (run this in Terminal 2)
streamlit run streamlit_app.py
```

Note: The Streamlit client uses the Python `requests` library to communicate via HTTP with the Django API.

---

## Demo Screenshots

![Streamlit RAG Demo Screenshot](assets/streamlit-demo-1.png)
![Streamlit RAG Demo Screenshot](assets/streamlit-demo-2.png)
![Streamlit RAG Demo Screenshot](assets/streamlit-demo-3.png)
![Full-Stack RAG Demo Screenshot](assets/fullstack-demo-1.png)
![Full-Stack RAG Demo Screenshot](assets/fullstack-demo-2.png)

---

## RAG API Endpoints (Django REST Framework)

The RAG core logic is exposed as a fully decoupled RESTful service.

```
Base URL: http://127.0.0.1:8000/api/v1/
```

---

### 1. Ingest Documents

Processes PDF files, chunks them, embeds them, and updates the local vector index.

```
Endpoint: /api/v1/ingest/
Method: POST
Content Type: multipart/form-data
Required Field: pdf_file (File)
```

---

### 2. Query Documents

Performs semantic retrieval and LLM answer generation.

```
Endpoint: /api/v1/query/
Method: POST
Content Type: application/json
Required Fields:
  - query (str)
Optional Fields:
  - k (int)
  - generate (bool)
```

---

## Qdrant (Dockerized Vector Database Integration)

An backend extension for persistent vector storage using **[Qdrant](https://qdrant.tech)** — a high-performance vector search engine.

This replaces the previous in-memory index with a Dockerized service that supports scalable, persistent embeddings and approximate nearest-neighbor search.

```bash
# 1. Run Qdrant in Docker
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_data:/qdrant/storage qdrant/qdrant

# 2. Run local Python test (verify connectivity)
python qdrant_try.py
```

Expected output:
```
upsert ok: {...}
search result: {...}
```

This confirms successful vector ingestion and retrieval on Windows + Docker within an isolated Python venv environment.


## Notes

```
- All operations (embedding, retrieval, generation) run fully locally.
- The vector index is stored under ./db/ and can be reused across sessions.
- The GGUF model must be downloaded separately and placed in the project root.
```
