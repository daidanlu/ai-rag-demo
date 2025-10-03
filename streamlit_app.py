import os
import streamlit as st
from pathlib import Path
from typing import Dict, Any

# Assumes rag.py contains ingest_files and answer functions
try:
    from rag import ingest_files, answer
except ImportError:
    st.error("Could not import ingest_files or answer functions from rag.py. Ensure rag.py exists and function names are correct.")
    st.stop()


# --- UI Configuration ---
# Using a standard, non-emoji page icon for a cleaner look
st.set_page_config(page_title="Local RAG Demo", page_icon="search", layout="centered")

st.title("Local RAG Demo (Local LLM)")
st.caption("Upload documents, build index, and query using a local language model.")

# --- Settings Expander ---
with st.expander("Configuration Settings", expanded=True):
    model = st.text_input(
        "LLM Model (.gguf filename or leave blank for default from rag.py)",
        value=""
    )
    # Using 'Top-K' for professional retrieval terminology
    k = st.slider("Retrieval Top-K Chunks", 1, 8, 4)

# --- 1) Upload and Ingestion ---
st.subheader("1) Document Upload and Ingestion")
uploaded = st.file_uploader("Drag and drop PDF files here", accept_multiple_files=True, type=["pdf"])

if uploaded:
    # Ensure the data directory exists
    data_dir = Path("data") 
    data_dir.mkdir(exist_ok=True, parents=True)
    
    saved = []
    # Save uploaded files to the data/ directory
    for f in uploaded:
        # Use f.name for cross-platform compatibility
        out = data_dir / f.name
        # Write file contents
        out.write_bytes(f.read())
        saved.append(str(out))
    
    st.info(f"Saved {len(saved)} file(s) to the `./data` directory")

    # Simple, professional button text
    if st.button("Ingest Documents"):
        # Use glob pattern to match all PDF files in data/
        n = ingest_files(["data/*.pdf"])
        # FIX: Replaced non-compliant 'check' shortcode with functional emoji '✅' 
        st.toast(f"Ingestion successful: Processed {n} chunks.", icon="✅")


# --- 2) Question Answering ---
st.subheader("2) Query Documents")
q = st.text_input("Enter your question here")

col1, col2 = st.columns([1,1])
with col1:
    # Changed button text to a more action-oriented verb
    run_btn = st.button("Generate Answer")
with col2:
    # Clean terminology for retrieval fallback
    dry = st.checkbox("Retrieve only (Skip LLM Generation)", value=False)

if run_btn and q.strip():
    with st.spinner('Processing request...'):
        kwargs: Dict[str, Any] = {"k": k}
        if model.strip():
            kwargs["model"] = model.strip() # Override default model if user specifies one
        
        # Call the answer function from rag.py
        res = answer(q.strip(), generate=not dry, **kwargs)

        st.markdown("---")
        
        # Display results
        answer_text = res.get("answer", "")
        if answer_text.startswith("[ERROR]"):
            # Handle LLM failure gracefully
            st.error(f"Answer generation failed. Check model configuration. Displaying raw retrieval results as fallback: {answer_text}")
            st.markdown("### Retrieved Chunks (Fallback)")
        else:
            st.markdown("### Generated Answer")
            st.markdown(answer_text) 

        st.markdown("### Retrieval Sources")
        
        hits = res.get("hits",[])
        if hits:
            # Iterate and display retrieved document chunks and metadata
            for i, h in enumerate(hits, start=1):
                meta = h.get("meta",{})
                distance_str = f"Distance={h.get('distance',None):.4f}" if h.get('distance') is not None else ""
                
                # Display metadata and distance
                st.write(f"**[{i}]** **Document:** {meta.get('doc_id','?')} • **Chunk Index:** {meta.get('chunk_idx','?')} • {distance_str}")
                
                # Show chunk content in an expander
                with st.expander(f"View Chunk {i} Content"):
                    st.code(h.get('text', 'N/A'), language="text")

        else:
             st.warning("No relevant sources retrieved. Please ensure documents have been ingested and the query is specific.")
