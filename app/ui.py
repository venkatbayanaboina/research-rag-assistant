import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import sys
import streamlit as st

# Include root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.components import render_sidebar, render_chat_interface

# Configure page settings
st.set_page_config(page_title="Multimodal Research RAG Suite", layout="wide")

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

# Premium CSS Injection for clean typography and gradients
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    /* Global Typography overrides */
    .stApp {
        font-family: 'Inter', sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.02em !important;
    }

    /* Linear Gradient Title */
    .app-title {
        background: linear-gradient(135deg, #a78bfa 0%, #3b82f6 50%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        font-family: 'Outfit', sans-serif;
    }
    
    .app-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 1.8rem;
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)

# Custom HTML Heading
st.markdown('<div class="app-title">Multimodal Research RAG Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Upload papers to index text in FAISS and layout elements in CLIP, then chat and compare visual figures side-by-side.</div>', unsafe_allow_html=True)


# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 1. Render Sidebar (returns settings and indexed document list)
use_rerank, indexed_docs = render_sidebar()

# 2. Render Main Chat Interface
render_chat_interface(use_rerank, indexed_docs)
