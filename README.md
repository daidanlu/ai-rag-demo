# AI RAG Demo (Local LLM Service)

A minimal, secure Retrieval-Augmented Generation (RAG) system built entirely from scratch in Python, focused on running inference and retrieval **locally and offline**. The core logic is exposed via a dedicated Django REST API service.

---

## Core Components

* **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
* **Vector Store:** Custom implementation using **NumPy** and `scikit-learn` (`NearestNeighbors`) for memory-based indexing (persisted in `./db`).
* **Generator:** Local LLM inference via **`llama-cpp-python`** (default model: `Llama-3.2-1B-Instruct-Q4_0.gguf`).

---

## Quickstart & Setup

### A. Backend Service (Django REST API)

This is the **core component** that handles ingestion, retrieval, and LLM generation.

1.  **Preparation:** Place your desired GGUF model file (e.g., `Llama-3.2-1B-Instruct-Q4_0.gguf`) in the project root directory.

2.  **Install Dependencies:** Install the Python Web framework and Django REST Framework.
    ```bash
    pip install -r requirements.txt
    pip install -r requirements-web.txt
    ```

3.  **Run Service:** Start the Django development server.
    ```bash
    python manage.py runserver
    ```

### B. Command Line Interface (CLI)

Used for direct backend testing or initial indexing.

```bash
# 1. Ingest PDF documents and build the index (index files saved to ./db)
python main.py ingest data/*.pdf

# 2. Ask a question based on your indexed documents
python main.py ask "What is the document about?"
````

-----

## RAG API Endpoints (Django REST Framework)

The RAG core logic is exposed as a fully decoupled RESTful service.

### Base URL

`http://127.0.0.1:8000/api/v1/`

### 1\. Ingest Documents

Processes PDF files, chunks them, embeds them, and updates the local vector index.

  - **Endpoint:** `/api/v1/ingest/`
  - **Method:** `POST`
  - **Content Type:** `multipart/form-data`
  - **Required Field:** `pdf_file` (File)
  - **Example Response (201 CREATED):**
    ```json
    {
        "status": "success",
        "message": "File ingested successfully.",
        "chunks_processed": 98,
        "document_id": "document_name.pdf"
    }
    ```

### 2\. Query Documents

Performs semantic retrieval and LLM answer generation.

  - **Endpoint:** `/api/v1/query/`
  - **Method:** `POST`
  - **Content Type:** `application/json`
  - **Required Fields:** `query` (str)
  - **Optional Fields:** `k` (int), `generate` (bool)
  - **Example Request Body:**
    ```json
    {
        "query": "What are the main military units mentioned in the document?",
        "k": 3,
        "generate": true
    }
    ```
  - **Example Response (200 OK):**
    ```json
    {
        "status": "success",
        "answer": "Artillery, infantry, and mounted units.",
        "sources": [
            { /* ... retrieval hit data ... */ }
        ]
    }
    ```

-----

## Web UI (Streamlit Frontend)

A simple, local web interface for interacting with the **Django Backend Service**.

### Setup and Run

1.  **Start Backend:** Ensure the Django service is running (see **A. Run Service** above).

2.  **Run Frontend:** Launch the Streamlit application.

    ```bash
    streamlit run streamlit_app.py
    ```

    *(Note: The Streamlit code must be updated to use the Python `requests` library to send HTTP requests to the Django API endpoints for full functionality.)*

### Demo Screenshots

![Streamlit RAG Demo Screenshot](assets/streamlit-demo-1.png)
![Streamlit RAG Demo Screenshot](assets/streamlit-demo-2.png)
![Streamlit RAG Demo Screenshot](assets/streamlit-demo-3.png)
