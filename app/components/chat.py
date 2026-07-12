import os
import sys
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.core.generator import execute_rag_pipeline

def render_chat_interface(use_rerank, indexed_docs):
    """
    Renders the unified Contextual Chat interface, Q&A,
    section summaries, and multi-document comparisons with active lock controls.
    """
    st.subheader("Chat with your Knowledge Base")
    
    # Initialize thinking / lock states in session_state
    if "thinking" not in st.session_state:
        st.session_state.thinking = False
    if "current_prompt" not in st.session_state:
        st.session_state.current_prompt = None

    # 1. Render Chat Message History (with diagrams)
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("images"):
                st.markdown("#### Retrieved Diagrams / Charts:")
                for img_path in msg["images"]:
                    if os.path.exists(img_path):
                        st.image(img_path)
                        
    # 2. Capture and route prompt inputs
    active_prompt = None
    
    # If not thinking, render the active input box
    if not st.session_state.thinking:
        if prompt := st.chat_input("Ask a question or request a summary (e.g., 'summarize attention-is-all-you-need-Paper.pdf')..."):
            st.session_state.thinking = True
            st.session_state.current_prompt = prompt
            st.rerun()
            
    # If currently processing, lock the input box and run pipeline
    if st.session_state.thinking and st.session_state.current_prompt:
        active_prompt = st.session_state.current_prompt
        st.chat_input("Gemini is processing your request...", disabled=True)
        
        # Display user bubble
        with st.chat_message("user"):
            st.markdown(active_prompt)
            
        # Append user message to history
        if not st.session_state.chat_history or st.session_state.chat_history[-1]["content"] != active_prompt:
            st.session_state.chat_history.append({"role": "user", "content": active_prompt})
            
        try:
            # Execute RAG Pipeline via the unified Core Orchestrator
            with st.spinner("Executing RAG pipeline..."):
                response = execute_rag_pipeline(
                    prompt=active_prompt,
                    indexed_docs=indexed_docs,
                    use_rerank=use_rerank,
                    chat_history=st.session_state.chat_history[:-1]
                )
                
            answer = response["answer"]
            image_results = response["image_results"]
            search_results = response["search_results"]
            is_special_intent = response["is_special_intent"]
                        
            # 3. Show model response and display image files
            with st.chat_message("assistant"):
                st.markdown(answer)
                saved_image_paths = []
                if image_results:
                    st.markdown("#### Retrieved Diagrams / Charts:")
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
            if not is_special_intent and search_results:
                with st.expander("View Retrieved Text Sources"):
                    for idx, result in enumerate(search_results):
                        chunk = result["chunk"]
                        st.markdown(f"**Source {idx+1}: {chunk['source_file']} (Page {chunk['page']})**")
                        st.text(chunk["text"])
                        st.markdown("---")
        finally:
            # Safely release the thinking lock and trigger rerun to re-enable inputs
            st.session_state.current_prompt = None
            st.session_state.thinking = False
            st.rerun()
            
    elif st.session_state.thinking:
        # Render locked input box if in thinking transition state
        st.chat_input("Gemini is processing your request...", disabled=True)
