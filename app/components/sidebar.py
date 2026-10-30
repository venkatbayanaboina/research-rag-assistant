import os
import sys
import streamlit as st
import threading
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.core.ingestion import parse_pdf
from src.core.chunker import process_text_chunks, process_image_chunks
from src.core.vector_store import add_document_to_store, get_indexed_documents

# Thread-safe progress dictionary to avoid Streamlit session context errors
PROGRESS_STATE = {
    "in_progress": False,
    "progress_val": 0.0,
    "status_msg": "",
    "error_msg": "",
    "success_msg": "",
    "cancel_requested": False
}

def bg_index_worker(file_path, strategy, uploaded_name, session_paths):
    try:
        PROGRESS_STATE["in_progress"] = True
        PROGRESS_STATE["error_msg"] = ""
        PROGRESS_STATE["success_msg"] = ""
        PROGRESS_STATE["cancel_requested"] = False

        # Stage 1: Parsing
        if PROGRESS_STATE["cancel_requested"]:
            raise InterruptedError("Ingestion cancelled by user.")

        PROGRESS_STATE["status_msg"] = f"Stage 1/3 · Parsing layout & figures ({strategy})"
        PROGRESS_STATE["progress_val"] = 0.15
        elements = parse_pdf(file_path, strategy=strategy)

        # Stage 2: Chunking
        if PROGRESS_STATE["cancel_requested"]:
            raise InterruptedError("Ingestion cancelled by user.")

        PROGRESS_STATE["status_msg"] = "Stage 2/3 · Chunking document elements"
        PROGRESS_STATE["progress_val"] = 0.50
        text_chunks = process_text_chunks(elements, file_path)

        if strategy == "hi_res":
            if PROGRESS_STATE["cancel_requested"]:
                raise InterruptedError("Ingestion cancelled by user.")
            image_chunks = process_image_chunks(elements, file_path)
        else:
            image_chunks = []

        # Stage 3: Embedding & Indexing
        if PROGRESS_STATE["cancel_requested"]:
            raise InterruptedError("Ingestion cancelled by user.")

        PROGRESS_STATE["status_msg"] = "Stage 3/3 · Embedding & indexing"
        PROGRESS_STATE["progress_val"] = 0.80
        add_document_to_store(text_chunks, image_chunks, session_paths=session_paths)

        # Complete
        PROGRESS_STATE["progress_val"] = 1.00
        PROGRESS_STATE["success_msg"] = f"Successfully indexed '{uploaded_name}'!"
    except InterruptedError as ie:
        PROGRESS_STATE["error_msg"] = str(ie)
        # Clean up partial upload files
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass
    except Exception as e:
        PROGRESS_STATE["error_msg"] = f"Error indexing '{uploaded_name}': {e}"
    finally:
        PROGRESS_STATE["in_progress"] = False


def _side_label(text, icon=""):
    st.markdown(
        f'<div class="side-label">{icon} {text}</div>'
        if icon else f'<div class="side-label">{text}</div>',
        unsafe_allow_html=True,
    )


def _doc_library():
    """Renders indexed documents as compact glass cards."""
    try:
        from src.core.vector_store import get_registry
        registry = get_registry()
    except Exception:
        registry = []

    indexed_docs = sorted({c.get("source_file") for c in registry if c.get("source_file")})
    if not indexed_docs:
        st.caption("Nothing here yet.")
        return []

    for doc in indexed_docs:
        doc_chunks = [c for c in registry if c.get("source_file") == doc]
        st.markdown(
            f'<div class="doc-card"><div class="doc-name">{doc}</div>'
            f'<div class="doc-meta">{len(doc_chunks)} chunks</div></div>',
            unsafe_allow_html=True,
        )
    return indexed_docs


def render_sidebar():
    """
    Renders the sidebar interface in Streamlit: upload & ingestion, the indexed
    library, and the reranking toggle.

    Destructive controls (per-document delete, Clear DB) are intentionally not
    exposed here — session stores are per-connection and auto-purged after 2h,
    and the backend functions remain available via CLI for deliberate cleanup.
    """
    strategy = config.DEFAULT_PARSE_STRATEGY

    with st.sidebar:
        _side_label("Upload")

        # 1. File Uploader
        if "uploader_key_suffix" not in st.session_state:
            st.session_state.uploader_key_suffix = 0

        uploader_key = f"pdf_uploader_{st.session_state.uploader_key_suffix}"

        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key=uploader_key,
            label_visibility="collapsed",
        )

        # Ingestion Runner
        if uploaded_files:
            button_disabled = PROGRESS_STATE["in_progress"]
            if st.button("Index Files", type="primary",
                         icon=":material/upload_file:", use_container_width=True,
                         disabled=button_disabled):
                PROGRESS_STATE["in_progress"] = True

                # Save files first
                from src.core.vector_store import get_paths
                paths = get_paths()
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(paths["raw_pdfs"], uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                # Capture static file names/paths before clearing the uploader widget
                file_info = [(os.path.join(paths["raw_pdfs"], f.name), f.name) for f in uploaded_files]

                def run_indexing_pipeline():
                    for f_path, f_name in file_info:
                        bg_index_worker(f_path, strategy, f_name, paths)

                thread = threading.Thread(target=run_indexing_pipeline)
                thread.start()

                # Increment the suffix to generate a fresh new uploader widget, clearing the UI queue
                st.session_state.uploader_key_suffix += 1
                st.rerun()

        # Render persistent indexing status (non-blocking, allows chatting simultaneously)
        @st.fragment(run_every="2s" if PROGRESS_STATE["in_progress"] else None)
        def render_progress_status():
            if PROGRESS_STATE["in_progress"]:
                st.progress(min(PROGRESS_STATE["progress_val"], 1.0))
                st.markdown(f'<div class="stage-msg">{PROGRESS_STATE["status_msg"]}</div>',
                            unsafe_allow_html=True)
                if st.button("Cancel", key="btn_cancel_ingest", icon=":material/cancel:",
                             use_container_width=True):
                    PROGRESS_STATE["cancel_requested"] = True
                    st.toast("Cancellation requested...")
                    st.rerun()
            else:
                # Safe state transition: Copy to session state & clear globals BEFORE calling st.rerun()
                # to prevent infinite recursive rerun loops.
                if PROGRESS_STATE["success_msg"]:
                    st.session_state.toast_msg = PROGRESS_STATE["success_msg"]
                    PROGRESS_STATE["success_msg"] = ""
                    st.rerun()
                if PROGRESS_STATE["error_msg"]:
                    st.session_state.err_msg = PROGRESS_STATE["error_msg"]
                    PROGRESS_STATE["error_msg"] = ""
                    st.rerun()

        render_progress_status()

        # Display feedback toasts and errors from session state to prevent looping
        if "toast_msg" in st.session_state and st.session_state.toast_msg:
            st.toast(st.session_state.toast_msg)
            st.session_state.toast_msg = ""

        if "err_msg" in st.session_state and st.session_state.err_msg:
            st.error(st.session_state.err_msg)
            st.session_state.err_msg = ""

        # 2. Indexed Library
        _side_label("Library")
        indexed_docs = _doc_library()

        if st.button("New Session", icon=":material/autorenew:", use_container_width=True,
                     help="Fresh empty library + cleared chat. Nothing is deleted: the old "
                          "session data is left for the 2-hour auto-purge."):
            import uuid
            st.session_state.session_id = uuid.uuid4().hex
            session_dir = os.path.join(config.STORAGE_DIR, "sessions", st.session_state.session_id)
            st.session_state.RAW_PDF_DIR = os.path.join(session_dir, "raw_pdfs")
            st.session_state.EXTRACTED_IMAGE_DIR = os.path.join(session_dir, "extracted_images")
            st.session_state.CHUNKS_PATH = os.path.join(session_dir, "chunks.json")
            st.session_state.IMAGES_REGISTRY_PATH = os.path.join(session_dir, "images_registry.json")
            st.session_state.TEXT_INDEX_PATH = os.path.join(session_dir, "text_db.faiss")
            st.session_state.IMAGE_INDEX_PATH = os.path.join(session_dir, "image_db.faiss")
            os.makedirs(st.session_state.RAW_PDF_DIR, exist_ok=True)
            os.makedirs(st.session_state.EXTRACTED_IMAGE_DIR, exist_ok=True)
            st.session_state.chat_history = []
            st.session_state.thinking = False
            st.session_state.current_prompt = None
            st.rerun()

    # Reranking is always on in the UI (it is measurably better and costs ~100ms);
    # config.RERANK_ENABLED remains available as a programmatic kill-switch.
    return config.RERANK_ENABLED, indexed_docs
