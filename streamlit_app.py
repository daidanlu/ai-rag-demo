import os
import streamlit as st
import requests
from pathlib import Path
from typing import Dict, Any

# replace RAG core methods with HTTP requests
API_BASE_URL = "http://127.0.0.1:8000/api/v1"
INGEST_URL = f"{API_BASE_URL}/ingest/"
QUERY_URL = f"{API_BASE_URL}/query/"
QUERY_RETRIEVE_URL = f"{API_BASE_URL}/query_retrieve/"

# try:
#    from rag import ingest_files, answer
# except ImportError:
#    st.error(
#        "Could not import ingest_files or answer functions from rag.py. Ensure rag.py exists and function names are correct."
#    )
#    st.stop()


# UI config
st.set_page_config(page_title="Local RAG Demo", page_icon="search", layout="centered")

st.title("Local RAG Demo (Full-Stack)")
st.caption(f"Frontend running on Streamlit, Backend on Django API ({API_BASE_URL}).")

# Health badge
try:
    health = requests.get(f"{API_BASE_URL}/health/", timeout=3).json()
    storage = health.get("storage", "unknown")
    alive = health.get("qdrant", {}).get("alive", None)
    badge = f"Backend: **{storage}**"
    if alive is True:
        badge += " • Qdrant: ✅"
    elif alive is False:
        badge += " • Qdrant: ❌"
    st.caption(badge)
except Exception:
    st.caption("Backend: unreachable")


# Settings Expander
with st.expander("Configuration Settings", expanded=True):
    model = st.text_input("LLM Model (.gguf filename - Read by Django)", value="")
    # Using 'Top-K' for professional retrieval terminology
    k = st.slider("Retrieval Top-K Chunks", 1, 8, 4)

# 1) Upload and Ingestion calling Django Ingest API
st.subheader("1) Document Upload and Ingestion")
uploaded_files = st.file_uploader(
    "Drag and drop PDF files here", accept_multiple_files=True, type=["pdf"]
)

if st.button("Ingest Documents"):
    if not uploaded_files:
        st.warning("Please upload one or more PDF files first.")
    else:
        # to ensure Django server is running
        try:
            with st.spinner("Uploading and Ingesting documents via Django API..."):
                total_chunks = 0

                for uploaded_file in uploaded_files:
                    # prepare fils in multipart/form-data formats
                    files = {
                        "pdf_file": (
                            uploaded_file.name,
                            uploaded_file,
                            "application/pdf",
                        )
                    }

                    # send POST request to Django Ingest API
                    response = requests.post(INGEST_URL, files=files)

                    if response.status_code == 201:
                        data = response.json()
                        total_chunks += data.get("chunks_processed", 0)
                        st.success(
                            f"Ingested '{uploaded_file.name}'. Processed {data.get('chunks_processed')} chunks."
                        )
                    else:
                        st.error(
                            f"Failed to ingest '{uploaded_file.name}'. Status: {response.status_code}. Detail: {response.text}"
                        )

                if total_chunks > 0:
                    st.toast(
                        f"Total Ingestion Complete: {total_chunks} chunks processed.",
                        icon="✅",
                    )

        except requests.exceptions.ConnectionError:
            st.error(
                f"Connection Error: Could not reach Django backend at {API_BASE_URL}. Ensure 'python manage.py runserver' is running."
            )
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")


# 2) Question Answering calling Django Query API
st.subheader("2) Query Documents")
q = st.text_input("Enter your question here")

col1, col2 = st.columns([1, 1])
with col1:
    run_btn = st.button("Generate Answer")
with col2:
    dry = st.checkbox("Retrieve only (Skip LLM Generation)", value=False)

if run_btn and q.strip():
    try:
        with st.spinner("Querying Django API and processing response..."):
            # prepare JSON Payload for user input
            payload = {
                "query": q.strip(),
                "k": k,
                "generate": not dry,
            }
            if model.strip():
                payload["model"] = model.strip()

            # send POST request to Django Query API, choose endpoint by 'dry' flag
            url = QUERY_RETRIEVE_URL if dry else QUERY_URL
            response = requests.post(url, json=payload)

            # check connection error
            if response.status_code == 500:
                # 500 could be LLM failure, parse JSON to get detailed info
                data = response.json()
                if "LLM generation failed" in data.get("message", ""):
                    st.error(
                        f"LLM Generation Failed (Check Model Config). Details: {data['details']}"
                    )
                    res = {
                        "answer": data.get("answer", ""),
                        "hits": data.get("sources", []),
                    }  # fallback here
                else:
                    raise Exception(
                        f"Backend 500 Error: {data.get('details', 'Unknown error')}"
                    )

            elif response.status_code != 200:
                st.error(
                    f"API Request Failed. Status: {response.status_code}. Detail: {response.text}"
                )
                st.stop()

            # success or after LLM failure 500 fallback
            data = response.json() if response.status_code == 200 else data
            res = {"answer": data.get("answer", ""), "hits": data.get("sources", [])}

            # display results
            st.markdown("---")
            st.markdown("### Generated Answer")
            st.markdown(res.get("answer", "No answer provided."))

            st.markdown("### Retrieval Sources")
            hits = res.get("hits", [])
            if hits:
                for i, h in enumerate(hits, start=1):
                    meta = h.get("meta", {})
                    distance_str = (
                        f"Distance={h.get('distance',None):.4f}"
                        if h.get("distance") is not None
                        else ""
                    )
                    st.write(
                        f"**[{i}]** **Document:** {meta.get('doc_id','?')} • **Chunk Index:** {meta.get('chunk_idx','?')} • {distance_str}"
                    )
                    with st.expander(f"View Chunk {i} Content"):
                        st.code(h.get("text", "N/A"), language="text")
            else:
                st.warning("No relevant sources retrieved.")

    except requests.exceptions.ConnectionError:
        st.error(
            f"Connection Error: Could not reach Django backend at {API_BASE_URL}. Ensure 'python manage.py runserver' is running."
        )
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
