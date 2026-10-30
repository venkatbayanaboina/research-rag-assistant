import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import sys
import streamlit as st

# Include root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.components import render_sidebar, render_chat_interface
from app.theme import inject_theme, hero_html

# Configure page settings
st.set_page_config(
    page_title="Multimodal Research RAG Suite",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

import uuid
import shutil
import time
import config

# Initialize dynamic user session isolation paths
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

    # Define session-specific storage paths
    session_dir = os.path.join(config.STORAGE_DIR, "sessions", st.session_state.session_id)
    st.session_state.RAW_PDF_DIR = os.path.join(session_dir, "raw_pdfs")
    st.session_state.EXTRACTED_IMAGE_DIR = os.path.join(session_dir, "extracted_images")
    st.session_state.CHUNKS_PATH = os.path.join(session_dir, "chunks.json")
    st.session_state.IMAGES_REGISTRY_PATH = os.path.join(session_dir, "images_registry.json")
    st.session_state.TEXT_INDEX_PATH = os.path.join(session_dir, "text_db.faiss")
    st.session_state.IMAGE_INDEX_PATH = os.path.join(session_dir, "image_db.faiss")

    # Ensure session directories exist
    os.makedirs(st.session_state.RAW_PDF_DIR, exist_ok=True)
    os.makedirs(st.session_state.EXTRACTED_IMAGE_DIR, exist_ok=True)

    # Auto-cleanup old inactive user sessions (older than 2 hours) on startup
    sessions_root = os.path.join(config.STORAGE_DIR, "sessions")
    if os.path.exists(sessions_root):
        for s_dir in os.listdir(sessions_root):
            full_path = os.path.join(sessions_root, s_dir)
            if os.path.isdir(full_path) and s_dir != st.session_state.session_id:
                try:
                    if time.time() - os.path.getmtime(full_path) > 7200:
                        shutil.rmtree(full_path)
                except:
                    pass

# Inject the design system before any content renders
inject_theme()

# Hero header
st.markdown(
    hero_html(
        "Multimodal Research RAG Suite",
        "Agentic, citation-verified answers over your own paper library.",
        ["Agentic pipeline", "Hybrid retrieval", "Cited answers"],
    ),
    unsafe_allow_html=True,
)

# Live library metrics (cheap JSON registry reads, session-aware)
try:
    from src.core.vector_store import get_registry, get_image_registry
    _registry = get_registry()
    _images = get_image_registry()
    n_docs = len({c.get("source_file") for c in _registry})
    n_chunks = len(_registry)
    n_figs = len(_images)
except Exception:
    n_docs = n_chunks = n_figs = 0

m1, m2, m3 = st.columns(3)
m1.metric("Documents", n_docs)
m2.metric("Chunks", f"{n_chunks:,}")
m3.metric("Figures", n_figs)

st.markdown("<div style='height: 4px'></div>", unsafe_allow_html=True)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 1. Render Sidebar (returns settings and indexed document list)
use_rerank, indexed_docs = render_sidebar()

# 2. Render Main Chat Interface
render_chat_interface(use_rerank, indexed_docs)
