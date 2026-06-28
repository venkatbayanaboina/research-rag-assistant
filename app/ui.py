import os
import sys
import streamlit as st

# Include root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.core.ingestion import parse_pdf
from src.core.chunker import process_chunks
from src.core.vector_store import get_indexed_documents, add_document_to_store, search_store, get_registry
from src.core.generator import generate_answer, generate_summary

# Configure page settings
st.set_page_config(page_title="Multi-PDF RAG Assistant", layout="wide", page_icon="📚")

st.title("📚 Modular Multi-PDF RAG Assistant")
st.write("Upload your research papers and documents to index them in FAISS, then summarize them or start a grounded chat session.")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for uploading and listing documents
with st.sidebar:
    st.header("📂 Document Control Center")
    
    # 1. File Uploader
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload one or multiple research papers"
    )
    
    if uploaded_files:
        if st.button("🚀 Process & Index Files", use_container_width=True):
            for uploaded_file in uploaded_files:
                file_path = os.path.join(config.RAW_PDF_DIR, uploaded_file.name)
                
                # Save the file locally
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                st.info(f"Processing '{uploaded_file.name}'...")
                
                try:
                    # Ingest -> Chunk -> Embed & FAISS Index
                    elements = parse_pdf(file_path, strategy="fast")
                    processed_chunks = process_chunks(elements, file_path)
                    add_document_to_store(processed_chunks)
                    st.success(f"Successfully indexed '{uploaded_file.name}'!")
                except Exception as e:
                    st.error(f"Error indexing '{uploaded_file.name}': {e}")
                    
            st.rerun()
            
    # 2. List of Indexed Files
    st.markdown("---")
    st.subheader("Indexed Documents")
    indexed_docs = get_indexed_documents()
    if indexed_docs:
        for doc in indexed_docs:
            st.markdown(f"- 📄 `{doc}`")
    else:
        st.caption("No documents indexed yet. Upload files to get started!")
        
    # 3. Actions
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
        st.rerun()

# Main Interface Tabs
tab1, tab2 = st.tabs(["💬 Contextual Chat", "📝 Document Summaries"])

# Tab 1: Chat with Documents
with tab1:
    st.subheader("Chat with your Knowledge Base")
    
    # Show chat messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Handle user message input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # 1. Show user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # 2. Search FAISS store
        with st.spinner("Searching vector database..."):
            search_results = search_store(prompt, k=5)
            
        # 3. Generate Answer
        with st.spinner("Gemini is thinking..."):
            try:
                answer = generate_answer(prompt, search_results, st.session_state.chat_history[:-1])
            except Exception as e:
                answer = f"Error generating answer: {e}"
                
        # 4. Show model response
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        # 5. Show sources in expander if results were retrieved
        if search_results:
            with st.expander("🔍 View Retrieved Sources"):
                for idx, result in enumerate(search_results):
                    chunk = result["chunk"]
                    st.markdown(f"**Source {idx+1}: {chunk['source_file']} (Page {chunk['page']})**")
                    st.text(chunk["text"])
                    st.markdown("---")

# Tab 2: Document Summaries
with tab2:
    st.subheader("Generate Executive Summaries")
    
    if indexed_docs:
        selected_doc = st.selectbox("Select document to summarize", indexed_docs)
        
        if st.button("📖 Generate Summary", type="primary"):
            with st.spinner("Extracting chunks & compiling summary with Gemini..."):
                try:
                    registry = get_registry()
                    # Filter chunks belonging to this document
                    doc_chunks = [chunk for chunk in registry if chunk["source_file"] == selected_doc]
                    
                    if doc_chunks:
                        summary = generate_summary(doc_chunks)
                        st.markdown("### Summary Results")
                        st.info(f"Summary for: **{selected_doc}**")
                        st.markdown(summary)
                    else:
                        st.error("No chunks found for the selected document.")
                except Exception as e:
                    st.error(f"Error generating summary: {e}")
    else:
        st.caption("No documents available. Please upload and index documents in the sidebar first.")
