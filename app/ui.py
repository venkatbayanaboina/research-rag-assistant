import os
import sys
import streamlit as st

# Include root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.components import render_sidebar, render_chat_interface

# Configure page settings
st.set_page_config(page_title="Multi-PDF Multimodal RAG Suite", layout="wide", page_icon="📚")

# Premium CSS Injection for sleek dark mode, custom typography, gradients, and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    /* Main App Container */
    .stApp {
        background-color: #0b0d16 !important;
        font-family: 'Inter', sans-serif !important;
        color: #e2e8f0 !important;
    }

    /* Target Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        color: #f8fafc !important;
        letter-spacing: -0.02em !important;
    }

    /* Linear Gradient Title */
    .app-title {
        background: linear-gradient(135deg, #a78bfa 0%, #3b82f6 50%, #10b981 100%);
        -webkit-background-clip: text;
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

    /* Sidebar Background & Border */
    section[data-testid="stSidebar"] {
        background-color: #111322 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    /* Sidebar Title */
    section[data-testid="stSidebar"] h2 {
        color: #a78bfa !important;
        font-size: 1.5rem !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    /* Customize Chat Message Container */
    div[data-testid="stChatMessage"] {
        background-color: #181b2e !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 14px !important;
        margin-bottom: 14px !important;
        padding: 20px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.25s ease-in-out !important;
    }
    div[data-testid="stChatMessage"]:hover {
        border-color: rgba(167, 139, 250, 0.25) !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.2) !important;
    }

    /* Custom Input Box styling */
    div[data-testid="stChatInput"] {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        background-color: #121424 !important;
    }

    /* Expanders styling */
    div[data-testid="stExpander"] {
        background-color: #14172a !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 10px !important;
    }

    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0b0d16;
    }
    ::-webkit-scrollbar-thumb {
        background: #272a44;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #3b82f6;
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
