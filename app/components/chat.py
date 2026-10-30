import os
import sys
import html as _html
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.core.generator import execute_rag_pipeline
from app.theme import verification_chip, source_head, source_text, figure_frame


def _empty_state(indexed_docs):
    """Branded welcome screen whose suggestion chips feed the normal pipeline."""
    st.markdown(
        """
<div class="empty-hero">
  <div class="empty-glyph">◈</div>
  <h2 class="empty-title">Ask your research library</h2>
  <p class="empty-sub">Grounded answers from your indexed papers — every claim checked against its source.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    if not indexed_docs:
        st.info("Upload a paper from the sidebar to get started.")
        return

    first_doc = indexed_docs[0]
    suggestions = [
        f"What are the key contributions of {first_doc.replace('.pdf', '')}?",
        "Build an evidence table for my papers",
        f"Summarize {first_doc}",
        "What are the research gaps across my papers?",
        "Show the main architecture figure from the most recent paper",
        "Do my papers contradict each other on their main findings?",
    ]
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                st.session_state.thinking = True
                st.session_state.current_prompt = suggestion
                st.rerun()


def render_chat_interface(use_rerank, indexed_docs):
    """
    Renders the unified Contextual Chat interface, Q&A,
    section summaries, and multi-document comparisons with active lock controls.
    """
    # Initialize thinking / lock states in session_state
    if "thinking" not in st.session_state:
        st.session_state.thinking = False
    if "current_prompt" not in st.session_state:
        st.session_state.current_prompt = None

    # 1. Render Chat Message History (with diagrams and collapsible references)
    for msg in st.session_state.chat_history:
        avatar = ":material/person:" if msg["role"] == "user" else ":material/smart_toy:"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            verification = msg.get("verification")
            if verification and verification.get("status") not in (None, "skipped"):
                st.markdown(
                    verification_chip(verification.get("status"),
                                      verification.get("citation_coverage")),
                    unsafe_allow_html=True,
                )
            if msg.get("images"):
                for img_data in msg["images"]:
                    if isinstance(img_data, dict):
                        st.markdown(figure_frame(img_data["path"], img_data.get("caption", "")),
                                    unsafe_allow_html=True)
                    elif os.path.exists(img_data):
                        st.image(img_data)
            if msg.get("sources"):
                with st.expander("Retrieved text sources", icon=":material/article:"):
                    for idx, src in enumerate(msg["sources"]):
                        st.markdown(source_head(src["file"], src["page"]),
                                    unsafe_allow_html=True)
                        st.markdown(source_text(src["text"]), unsafe_allow_html=True)
            if msg.get("images"):
                with st.expander("Retrieved visual sources", icon=":material/image_search:"):
                    for idx, img_data in enumerate(msg["images"]):
                        if isinstance(img_data, dict):
                            st.markdown(figure_frame(img_data["path"],
                                                     f"Visual source {idx+1} · {img_data.get('caption', '')}"),
                                        unsafe_allow_html=True)
                        elif os.path.exists(img_data):
                            st.image(img_data)

    # Empty state with actionable suggestions
    if not st.session_state.chat_history and not st.session_state.thinking:
        _empty_state(indexed_docs)

    # 2. Capture and route prompt inputs
    active_prompt = None

    # If not thinking, render the active input box
    if not st.session_state.thinking:
        if prompt := st.chat_input("Ask about your papers — e.g. 'compare the two most recent papers'..."):
            st.session_state.thinking = True
            st.session_state.current_prompt = prompt
            st.rerun()

    # If currently processing, lock the input box and run pipeline
    if st.session_state.thinking and st.session_state.current_prompt:
        active_prompt = st.session_state.current_prompt
        st.chat_input("Pipeline running — retrieving evidence and verifying citations…", disabled=True)

        # Display user bubble
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(active_prompt)

        # Append user message to history
        if not st.session_state.chat_history or st.session_state.chat_history[-1]["content"] != active_prompt:
            st.session_state.chat_history.append({"role": "user", "content": active_prompt})

        try:
            # Execute RAG Pipeline via the unified Core Orchestrator
            with st.spinner("Routing intent → hybrid retrieval → reranking → generation…"):
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
            verification = response.get("verification")
            actions = response.get("actions") or []

            # 3. Show model response and display image files
            with st.chat_message("assistant", avatar=":material/smart_toy:"):
                st.markdown(answer)
                saved_images = []
                if image_results:
                    for img_res in image_results[:1]:
                        img_chunk = img_res["chunk"]
                        path = img_chunk["image_path"]
                        if os.path.exists(path):
                            caption_str = (f"{img_chunk['source_file']} · Page {img_chunk['page']}"
                                           f" · relevance {img_res['score']:.3f}")
                            st.markdown(figure_frame(path, caption_str), unsafe_allow_html=True)
                            saved_images.append({
                                "path": path,
                                "caption": caption_str
                            })

                # Display active sources expander locally during creation
                if not is_special_intent and search_results:
                    with st.expander("Retrieved text sources", icon=":material/article:",
                                     expanded=False):
                        for idx, result in enumerate(search_results):
                            chunk = result["chunk"]
                            st.markdown(source_head(chunk["source_file"], chunk["page"]),
                                        unsafe_allow_html=True)
                            st.markdown(source_text(chunk["text"]), unsafe_allow_html=True)
                if not is_special_intent and saved_images:
                    with st.expander("Retrieved visual sources", icon=":material/image_search:"):
                        for idx, img_data in enumerate(saved_images):
                            if os.path.exists(img_data["path"]):
                                st.markdown(figure_frame(img_data["path"], img_data["caption"]),
                                            unsafe_allow_html=True)

                # Agent action log (research intents show the tools that ran)
                if actions:
                    with st.expander("Agent actions", icon=":material/bolt:"):
                        for a in actions:
                            line = f"`{a.get('tool')}`"
                            if a.get("args"):
                                line += f" · {a['args']}"
                            if a.get("summary"):
                                line += f" — {a['summary']}"
                            if a.get("skipped"):
                                line += f" — *skipped: {a['skipped']}*"
                            if a.get("error"):
                                line += f" — ⚠️ {a['error']}"
                            st.markdown(line)

                # Verification badge chip
                if not is_special_intent and verification and \
                        verification.get("status") not in (None, "skipped"):
                    st.markdown(
                        verification_chip(verification.get("status"),
                                          verification.get("citation_coverage")),
                        unsafe_allow_html=True,
                    )

            # Convert retrieved text search results into history source objects
            saved_sources = []
            if not is_special_intent and search_results:
                for res in search_results:
                    chunk = res["chunk"]
                    saved_sources.append({
                        "file": chunk["source_file"],
                        "page": chunk["page"],
                        "text": chunk["text"]
                    })

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "images": saved_images,
                "sources": saved_sources,
                "verification": verification if not is_special_intent else None
            })
        finally:
            # Safely release the thinking lock and trigger rerun to re-enable inputs
            st.session_state.current_prompt = None
            st.session_state.thinking = False
            st.rerun()

    elif st.session_state.thinking:
        # Render locked input box if in thinking transition state
        st.chat_input("Pipeline running — retrieving evidence and verifying citations…", disabled=True)
