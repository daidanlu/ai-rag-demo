import os
import time
import streamlit as st
import requests
from streamlit.components.v1 import html as st_html
from pathlib import Path
from typing import Dict, Any

# replace RAG core methods with HTTP requests
API_BASE_URL = "http://127.0.0.1:8000/api/v1"
INGEST_URL = f"{API_BASE_URL}/ingest/"
QUERY_URL = f"{API_BASE_URL}/query/"
QUERY_RETRIEVE_URL = f"{API_BASE_URL}/query_retrieve/"


# backend clear helper: Call Django /api/v1/clear/ and return JSON result.
def clear_index_backend() -> dict:
    r = requests.post(f"{API_BASE_URL}/clear/", timeout=30)
    r.raise_for_status()
    return r.json()


# try:
#    from rag import ingest_files, answer
# except ImportError:
#    st.error(
#        "Could not import ingest_files or answer functions from rag.py. Ensure rag.py exists and function names are correct."
#    )
#    st.stop()

# UI header
st.title("Local RAG Demo (Full-Stack)")
st.caption(f"Frontend running on Streamlit, Backend on Django API ({API_BASE_URL}).")
st.caption(f"Streamlit version: {st.__version__}")

# Health banner
colA, colB = st.columns([1, 1])

with colA:
    health_box = st.empty()

    def render_health_once():
        """Fetch /health and render one-line badge into health_box."""
        try:
            # quick retries for first-load， add ts + no-cache to avoid caching
            j = None
            for _ in range(3):
                try:
                    r = requests.get(
                        f"{API_BASE_URL}/health/?_={int(time.time())}",
                        headers={"Cache-Control": "no-cache"},
                        timeout=3,
                    )
                    j = r.json()
                    break
                except Exception:
                    time.sleep(0.4)
            if not isinstance(j, dict):
                raise RuntimeError("health fetch failed")

            storage = j.get("storage", "unknown")
            alive = (j.get("qdrant") or {}).get("alive", None)
            badge = f"Backend: **{storage}**"
            if alive is True:
                badge += " • Qdrant: ✅"
            elif alive is False:
                badge += " • Qdrant: ❌"
            else:
                badge += " • Qdrant: ⚪"
            points = (j.get("qdrant") or {}).get("points_count", None)
            if points is not None:
                badge += f" • vectors: {points}"
            latency = j.get("latency_ms", None)
            if latency is not None:
                badge += f" • latency: {latency} ms"
            health_box.caption(badge)
        except Exception:
            health_box.caption("Backend: unreachable")

    # render once per run
    render_health_once()

with colB:
    auto = st.toggle(
        "Auto-refresh health (10s)",
        value=False,
        key="health_toggle_nonblocking",
        help="Auto-refresh the health banner without blocking the page.",
    )


# schedule non-blocking reruns every 10s (pause when busy)
busy = st.session_state.get("busy", False)
if auto and not busy:
    st.markdown("<meta http-equiv='refresh' content='10'/>", unsafe_allow_html=True)


# Settings Expander
with st.expander("Configuration Settings", expanded=True):
    model = st.text_input("LLM Model (.gguf filename - Read by Django)", value="")
    # Using 'Top-K' for professional retrieval terminology
    k = st.slider("Retrieval Top-K Chunks", 1, 8, 4)


# Maintenance section: clear index button
st.divider()
st.subheader("Maintenance")

with st.container(border=True):
    st.caption("Clear the vector index on the backend (Qdrant or in-memory).")

    busy = st.session_state.get("busy", False)

    col1, col2 = st.columns([1, 3])
    with col1:
        confirm = st.checkbox("I'm sure", key="confirm_clear_index")
    with col2:
        clear_btn = st.button(
            "Clear Index (backend)",
            type="primary",
            disabled=(not confirm) or busy,
            help="Drop & recreate the collection in Qdrant (or reset memory store).",
        )

    if clear_btn:
        st.session_state["busy"] = True  # pause autorefresh
        try:
            with st.spinner("Clearing index..."):
                result = clear_index_backend()
            ok = bool(result.get("result", {}).get("ok"))
            if ok:
                st.success("Index cleared successfully.")
            else:
                st.warning(f"Clear API returned: {result}")
        except requests.HTTPError as e:
            st.error(f"HTTP error: {e.response.status_code} — {e.response.text[:300]}")
        except Exception as e:
            st.error(f"Failed to clear index: {e}")
        finally:
            st.session_state["busy"] = False
            st.rerun()  # immediate light refresh to update health

# 1) Upload and Ingestion calling Django Ingest API
st.subheader("1) Document Upload and Ingestion")
uploaded_files = st.file_uploader(
    "Upload PDF(s)", accept_multiple_files=True, type=["pdf"]
)

busy = st.session_state.get("busy", False)
ingest_btn = st.button(
    "Ingest Documents",
    type="primary",
    disabled=busy or not uploaded_files,
    help="Upload selected PDF(s) to the Django backend for ingestion.",
)

if ingest_btn:
    # to ensure Django server is running
    st.session_state["busy"] = True  # to prevent interruptions of requests
    try:
        total = len(uploaded_files)
        total_chunks = 0
        ok_cnt = 0
        fail_cnt = 0

        progress = st.progress(0, text=f"Uploading 0/{total}")

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            with st.spinner(f"Uploading {uploaded_file.name} ({idx}/{total})..."):

                file_bytes = uploaded_file.read()
                files = {
                    "pdf_file": (
                        uploaded_file.name,
                        file_bytes,
                        "application/pdf",
                    )
                }

                response = requests.post(INGEST_URL, files=files, timeout=120)

            if response.status_code in (200, 201):
                try:
                    data = response.json()
                except Exception:
                    data = {}
                ok_cnt += 1
                chunks = int(data.get("chunks_processed", 0))
                total_chunks += chunks
                st.toast(
                    f"✅ {uploaded_file.name} — processed {chunks} chunks.", icon="✅"
                )
            else:
                fail_cnt += 1
                st.toast(
                    f"❌ {uploaded_file.name} — {response.status_code}: {response.text[:200]}",
                    icon="❌",
                )

            progress.progress(idx / total, text=f"Uploading {idx}/{total}")

        if ok_cnt > 0:
            st.success(
                f"Completed: {ok_cnt}/{total} file(s) ingested, total {total_chunks} chunks."
            )
        if fail_cnt > 0:
            st.warning(f"{fail_cnt}/{total} file(s) failed.")

        # give toast/success messages time to be visible
        time.sleep(2)
        # rerun to update，health vectors
        st.rerun()

    except requests.exceptions.ConnectionError:
        st.error(
            f"Connection Error: Could not reach Django backend at {API_BASE_URL}. Ensure 'python manage.py runserver' is running."
        )
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    finally:
        st.session_state["busy"] = False


# 2) Question Answering calling Django Query API
st.subheader("2) Query Documents")
q = st.text_input("Enter your question here")

col1, col2 = st.columns([1, 1])
with col1:
    run_btn = st.button("Generate Answer")
with col2:
    dry = st.checkbox("Retrieve only (Skip LLM Generation)", value=False)

show_retrieval = st.toggle(
    "Show retrieval details",
    value=False,
    help="Display top-K retrieved chunks with scores and source metadata.",
)

# Max tokens slider (controls answer length)
max_tokens = st.slider(
    "Max tokens for answer",
    min_value=200,
    max_value=1500,
    value=500,
    step=100,
    help="Upper bound for generated answer length; increase if answers are cut off.",
)

if run_btn and q.strip():
    try:
        with st.spinner("Querying Django API and processing response..."):
            start_ts = time.time()  # timing starts
            # prepare JSON Payload for user input
            payload = {
                "query": q.strip(),
                "k": k,
                "generate": not dry,
            }

            if model.strip():
                payload["model"] = model.strip()

            payload["max_tokens"] = int(max_tokens)

            # send POST request to Django Query API, choose endpoint by 'dry' flag
            url = QUERY_RETRIEVE_URL if dry else QUERY_URL
            response = requests.post(url, json=payload, timeout=900)

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
            res = {
                "answer": data.get("answer", ""),
                "hits": data.get("sources", []) or data.get("hits", []),
                "used": data.get("used", {}),
            }
            elapsed_ms = round((time.time() - start_ts) * 1000, 1)  # timing ends

            # display results
            st.markdown("---")
            st.markdown("### Generated Answer")
            st.caption(f"Answer latency: {elapsed_ms} ms")
            answer_text = res.get("answer", "No answer provided.")
            # to show full answer of long text, preventing trucations by Markdown
            st.text_area(
                "Full Answer Output",
                value=answer_text,
                height=300,
                disabled=True,
                key="answer_box",
            )

            if show_retrieval:
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
