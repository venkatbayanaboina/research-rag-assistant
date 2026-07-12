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

def bg_index_worker(file_path, strategy, uploaded_name):
    try:
        PROGRESS_STATE["in_progress"] = True
        PROGRESS_STATE["error_msg"] = ""
        PROGRESS_STATE["success_msg"] = ""
        PROGRESS_STATE["cancel_requested"] = False
        
        # Stage 1: Parsing
        if PROGRESS_STATE["cancel_requested"]:
            raise InterruptedError("Ingestion cancelled by user.")
            
        PROGRESS_STATE["status_msg"] = f"Stage 1/3: Parsing PDF layout & structures (strategy: {strategy})..."
        PROGRESS_STATE["progress_val"] = 0.15
        elements = parse_pdf(file_path, strategy=strategy)
        
        # Stage 2: Chunking
        if PROGRESS_STATE["cancel_requested"]:
            raise InterruptedError("Ingestion cancelled by user.")
            
        PROGRESS_STATE["status_msg"] = "Stage 2/3: Chunking document elements..."
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
            
        PROGRESS_STATE["status_msg"] = "Stage 3/3: Generating embeddings and indexing in FAISS..."
        PROGRESS_STATE["progress_val"] = 0.80
        add_document_to_store(text_chunks, image_chunks)
        
        # Complete
        PROGRESS_STATE["progress_val"] = 1.00
        PROGRESS_STATE["success_msg"] = f"Successfully indexed '{uploaded_name}'!"
        PROGRESS_STATE["in_progress"] = False
    except InterruptedError as ie:
        PROGRESS_STATE["error_msg"] = str(ie)
        PROGRESS_STATE["in_progress"] = False
        # Clean up partial upload files
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass
    except Exception as e:
        PROGRESS_STATE["error_msg"] = f"Error indexing '{uploaded_name}': {e}"
        PROGRESS_STATE["in_progress"] = False

def render_sidebar():
    """
    Renders the sidebar interface in Streamlit, including Strategy options,
    File upload controls, Background progress bar, and Reranking toggles.
    Returns:
        use_rerank (bool): True if Rerank toggle is enabled
        indexed_docs (list): List of currently indexed files
    """
    with st.sidebar:
        st.header("Document Control Center")
        
        # 1. Parsing Strategy Choice
        strategy = st.selectbox(
            "PDF Parsing Strategy",
            options=["fast", "hi_res"],
            index=0,
            help="fast: text only (very fast). hi_res: extracts diagrams/tables (requires layout parsing, runs locally)."
        )
        
        # 2. File Uploader
        uploaded_files = st.file_uploader(
            "Upload PDF files", 
            type=["pdf"], 
            accept_multiple_files=True,
            help="Upload one or multiple research papers"
        )
        
        # Ingestion Runner
        if uploaded_files:
            button_disabled = PROGRESS_STATE["in_progress"]
            if st.button("Process & Index Files", use_container_width=True, disabled=button_disabled):
                PROGRESS_STATE["in_progress"] = True
                
                # Save files first
                from src.core.vector_store import get_paths
                paths = get_paths()
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(paths["raw_pdfs"], uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                def run_indexing_pipeline():
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(paths["raw_pdfs"], uploaded_file.name)
                        bg_index_worker(file_path, strategy, uploaded_file.name)
                        
                thread = threading.Thread(target=run_indexing_pipeline)
                thread.start()
                st.rerun()

        # Render persistent indexing status (non-blocking, allows chatting simultaneously)
        if PROGRESS_STATE["in_progress"]:
            st.markdown("---")
            st.markdown("**Background Ingestion Active**")
            st.info(PROGRESS_STATE["status_msg"])
            if st.button("Cancel Ingestion", key="btn_cancel_ingest", use_container_width=True):
                PROGRESS_STATE["cancel_requested"] = True
                st.toast("Cancellation requested...")
                st.rerun()
            
        if PROGRESS_STATE["success_msg"]:
            st.toast(PROGRESS_STATE["success_msg"])
            PROGRESS_STATE["success_msg"] = ""
            
        if PROGRESS_STATE["error_msg"]:
            st.error(PROGRESS_STATE["error_msg"])
            PROGRESS_STATE["error_msg"] = ""
            
        # 3. List of Indexed Files
        st.markdown("---")
        st.subheader("Indexed Documents")
        indexed_docs = get_indexed_documents()
        if indexed_docs:
            for doc in indexed_docs:
                col_name, col_btn = st.columns([0.75, 0.25])
                with col_name:
                    st.markdown(f"`{doc}`")
                with col_btn:
                    if st.button("Delete", key=f"del_{doc}", use_container_width=True):
                        from src.core.vector_store import delete_document_from_store
                        delete_document_from_store(doc)
                        st.toast(f"Successfully purged '{doc}'!")
                        st.rerun()
        else:
            st.caption("No documents indexed yet. Upload files to get started!")
            
        # 4. Actions
        st.markdown("---")
        use_rerank = st.toggle("Enable Reranking (Cross-Encoder)", value=config.RERANK_ENABLED)
        
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
            
    return use_rerank, indexed_docs
