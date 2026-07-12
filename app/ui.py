import os
import sys
import streamlit as st

# Include root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.components import render_sidebar, render_chat_interface

# Configure page settings
st.set_page_config(page_title="Multi-PDF Multimodal RAG Suite", layout="wide", page_icon="📚")

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
st.markdown('<div class="app-title">📚 Multimodal Research RAG Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Upload papers to index text in FAISS and layout elements in CLIP, then chat and compare visual figures side-by-side.</div>', unsafe_allow_html=True)


# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 1. Render Sidebar (returns settings and indexed document list)
use_rerank, indexed_docs = render_sidebar()

# 2. Render Main Chat Interface
render_chat_interface(use_rerank, indexed_docs)
