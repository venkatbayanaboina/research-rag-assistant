import os
import sys
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.core.vector_store import search_store, search_image_store, get_registry
from src.core.generator import generate_answer, generate_summary, generate_section_summaries, generate_comparison
from app.components.router import detect_summary_request, detect_comparison_request

def render_chat_interface(use_rerank, indexed_docs):
    """
    Renders the unified Contextual Chat interface, including Q&A,
    section summaries, and multi-document comparisons.
    """
    st.subheader("Chat with your Knowledge Base")
    
    # 1. Render Chat Message History (with diagrams)
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("images"):
                st.markdown("#### 🖼️ Retrieved Diagrams / Charts:")
                for img_path in msg["images"]:
                    if os.path.exists(img_path):
                        st.image(img_path)
                        
    # 2. Handle new Chat Input
    if prompt := st.chat_input("Ask a question or request a summary (e.g., 'summarize attention-is-all-you-need-Paper.pdf')..."):
        # Display user bubble
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Route based on detected NLP intent
        comparison_targets = detect_comparison_request(prompt, indexed_docs)
        target_doc, is_section_wise = detect_summary_request(prompt, indexed_docs)
        
        if len(comparison_targets) >= 2:
            # Multi-Paper Comparison
            doc_a, doc_b = comparison_targets[0], comparison_targets[1]
            with st.spinner(f"Retrieving chunks & generating comparison matrix between '{doc_a}' and '{doc_b}'..."):
                try:
                    registry = get_registry()
                    doc_a_chunks = [chunk for chunk in registry if chunk["source_file"] == doc_a]
                    doc_b_chunks = [chunk for chunk in registry if chunk["source_file"] == doc_b]
                    
                    if doc_a_chunks and doc_b_chunks:
                        answer = generate_comparison(doc_a, doc_a_chunks, doc_b, doc_b_chunks)
                    else:
                        answer = f"Error: Could not retrieve chunks for both '{doc_a}' and '{doc_b}'."
                except Exception as e:
                    answer = f"Error generating comparison: {e}"
            image_results = None
            is_special_intent = True
            
        elif target_doc:
            # Document Summary Request
            summary_type_name = "section-wise summary" if is_section_wise else "executive summary"
            with st.spinner(f"Extracting chunks & compiling {summary_type_name} for '{target_doc}'..."):
                try:
                    registry = get_registry()
                    doc_chunks = [chunk for chunk in registry if chunk["source_file"] == target_doc]
                    
                    if doc_chunks:
                        if is_section_wise:
                            answer = generate_section_summaries(doc_chunks)
                        else:
                            answer = generate_summary(doc_chunks)
                    else:
                        answer = f"Error: No text chunks found in registry for document '{target_doc}'."
                except Exception as e:
                    answer = f"Error generating summary: {e}"
            image_results = None
            is_special_intent = True
            
        else:
            # Standard RAG Q&A
            is_special_intent = False
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
        if not is_special_intent and 'search_results' in locals() and search_results:
            with st.expander("🔍 View Retrieved Text Sources"):
                for idx, result in enumerate(search_results):
                    chunk = result["chunk"]
                    st.markdown(f"**Source {idx+1}: {chunk['source_file']} (Page {chunk['page']})**")
                    st.text(chunk["text"])
                    st.markdown("---")
