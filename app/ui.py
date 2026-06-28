import os
import sys
import streamlit as st
import threading

# Global progress state dictionary to avoid Streamlit thread context errors
PROGRESS_STATE = {
    "in_progress": False,
    "progress_val": 0.0,
    "status_msg": "",
    "error_msg": "",
    "success_msg": ""
}

# Include root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.core.ingestion import parse_pdf
from src.core.chunker import process_text_chunks, process_image_chunks
from src.core.vector_store import (
    get_indexed_documents,
    add_document_to_store,
    search_store,
    search_image_store,
    get_registry
)
from src.core.generator import generate_answer, generate_summary

def bg_index_worker(file_path, strategy, uploaded_name):
    try:
        PROGRESS_STATE["in_progress"] = True
        PROGRESS_STATE["error_msg"] = ""
        PROGRESS_STATE["success_msg"] = ""
        
        # Stage 1: Parsing
        PROGRESS_STATE["status_msg"] = f"Stage 1/3: Parsing PDF layout & structures (strategy: {strategy})..."
        PROGRESS_STATE["progress_val"] = 0.15
        elements = parse_pdf(file_path, strategy=strategy)
        
        # Stage 2: Chunking
        PROGRESS_STATE["status_msg"] = "Stage 2/3: Chunking document elements..."
        PROGRESS_STATE["progress_val"] = 0.50
        text_chunks = process_text_chunks(elements, file_path)
        image_chunks = process_image_chunks(elements, file_path) if strategy == "hi_res" else []
        
        # Stage 3: Embedding & Indexing
        PROGRESS_STATE["status_msg"] = "Stage 3/3: Generating embeddings and indexing in FAISS..."
        PROGRESS_STATE["progress_val"] = 0.80
        add_document_to_store(text_chunks, image_chunks)
        
        # Complete
        PROGRESS_STATE["progress_val"] = 1.00
        PROGRESS_STATE["success_msg"] = f"Successfully indexed '{uploaded_name}'!"
        PROGRESS_STATE["in_progress"] = False
    except Exception as e:
        PROGRESS_STATE["error_msg"] = f"Error indexing '{uploaded_name}': {e}"
        PROGRESS_STATE["in_progress"] = False

def detect_summary_request(prompt, indexed_files):
    """
    Parses the prompt to see if it's a request to summarize a document.
    Returns the target filename if matched, otherwise None.
    """
    p_lower = prompt.lower()
    keywords = ["summarize", "summary", "summarization", "executive summary", "summarise"]
    
    # Check if any summary keywords are present
    has_keyword = any(kw in p_lower for kw in keywords)
    if not has_keyword:
        return None
        
    if not indexed_files:
        return None
        
    # Attempt 1: Look for exact or partial matches of indexed filenames in the prompt
    for filename in indexed_files:
        name_only = os.path.splitext(filename)[0].lower()
        if filename.lower() in p_lower or name_only in p_lower:
            return filename
            
    # Attempt 2: Default to the most recently indexed file if keywords exist but no name matches
    return indexed_files[-1]

# Configure page settings
st.set_page_config(page_title="Multi-PDF RAG Assistant", layout="wide", page_icon="📚")

st.title("📚 Modular Multi-PDF Multimodal RAG Assistant")
st.write("Upload your research papers to index text in FAISS and visual elements in CLIP, then chat or generate summaries.")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for uploading and listing documents
with st.sidebar:
    st.header("📂 Document Control Center")
    
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
    
    if uploaded_files:
        button_disabled = PROGRESS_STATE["in_progress"]
        if st.button("🚀 Process & Index Files", use_container_width=True, disabled=button_disabled):
            PROGRESS_STATE["in_progress"] = True
            
            # Save files first
            for uploaded_file in uploaded_files:
                file_path = os.path.join(config.RAW_PDF_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            def run_indexing_pipeline():
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(config.RAW_PDF_DIR, uploaded_file.name)
                    bg_index_worker(file_path, strategy, uploaded_file.name)
                    
            thread = threading.Thread(target=run_indexing_pipeline)
            thread.start()
            st.rerun()

    # Render persistent indexing status
    if PROGRESS_STATE["in_progress"]:
        st.markdown("---")
        st.markdown("⏳ **Background Indexing Active**")
        st.progress(PROGRESS_STATE["progress_val"])
        st.caption(PROGRESS_STATE["status_msg"])
        
        import time
        time.sleep(0.5)
        st.rerun()
        
    if PROGRESS_STATE["success_msg"]:
        st.success(PROGRESS_STATE["success_msg"])
        PROGRESS_STATE["success_msg"] = ""
        st.rerun()
        
    if PROGRESS_STATE["error_msg"]:
        st.error(PROGRESS_STATE["error_msg"])
        PROGRESS_STATE["error_msg"] = ""
        st.rerun()
            
    # 3. List of Indexed Files
    st.markdown("---")
    st.subheader("Indexed Documents")
    indexed_docs = get_indexed_documents()
    if indexed_docs:
        for doc in indexed_docs:
            st.markdown(f"- 📄 `{doc}`")
    else:
        st.caption("No documents indexed yet. Upload files to get started!")
        
    # 4. Actions
    st.markdown("---")
    use_rerank = st.toggle("Enable Reranking (Cross-Encoder)", value=config.RERANK_ENABLED)
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
        st.rerun()

# Main Interface Chat
st.subheader("Chat with your Knowledge Base")

# Show chat messages (including persisted images)
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("images"):
            st.markdown("#### 🖼️ Retrieved Diagrams / Charts:")
            for img_path in msg["images"]:
                if os.path.exists(img_path):
                    st.image(img_path)
        
# Handle user message input
if prompt := st.chat_input("Ask a question or request a summary (e.g., 'summarize attention-is-all-you-need-Paper.pdf')..."):
    # 1. Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # 2. Check if this is a summary request
    target_doc = detect_summary_request(prompt, indexed_docs)
    
    if target_doc:
        # Generate Document Summary
        with st.spinner(f"Extracting chunks & compiling summary with Gemini for '{target_doc}'..."):
            try:
                registry = get_registry()
                doc_chunks = [chunk for chunk in registry if chunk["source_file"] == target_doc]
                
                if doc_chunks:
                    answer = generate_summary(doc_chunks)
                else:
                    answer = f"Error: No text chunks found in registry for document '{target_doc}'."
            except Exception as e:
                answer = f"Error generating summary: {e}"
        image_results = None
    else:
        # Run Standard RAG retrieval, reranking, and generation pipeline
        with st.spinner("Searching text database..."):
            search_results = search_store(prompt, rerank=use_rerank)
            
        with st.spinner("Searching visual database with CLIP..."):
            image_results = search_image_store(prompt)
            
        with st.spinner("Gemini is thinking..."):
            try:
                answer = generate_answer(prompt, search_results, image_results, st.session_state.chat_history[:-1])
            except Exception as e:
                answer = f"Error generating answer: {e}"
                
    # 3. Show model response and display image files
    with st.chat_message("assistant"):
        st.markdown(answer)
        saved_image_paths = []
        if image_results:
            st.markdown("#### 🖼️ Retrieved Diagrams / Charts:")
            for img_res in image_results:
                img_chunk = img_res["chunk"]
                path = img_chunk["image_path"]
                if os.path.exists(path):
                    st.image(path, caption=f"Source: {img_chunk['source_file']} (Page {img_chunk['page']}) - CLIP Score: {img_res['score']:.4f}")
                    saved_image_paths.append(path)
                    
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "images": saved_image_paths
    })
    
    # 4. Show text sources in expander if standard RAG search results were retrieved
    if not target_doc and 'search_results' in locals() and search_results:
        with st.expander("🔍 View Retrieved Text Sources"):
            for idx, result in enumerate(search_results):
                chunk = result["chunk"]
                st.markdown(f"**Source {idx+1}: {chunk['source_file']} (Page {chunk['page']})**")
                st.text(chunk["text"])
                st.markdown("---")
