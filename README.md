# AI RAG Demo (Local with GPT4All)

A minimal Retrieval-Augmented Generation (RAG) system:
- Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- Vector store: Chroma (persisted in ./db)
- Generator: GPT4All (runs locally)

## Quickstart
```bash
python main.py ingest data/*.pdf
python main.py ask "What is the document about?"
