import os
import sys
import streamlit as st

# Include root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.components import render_sidebar, render_chat_interface

# Configure page settings
st.set_page_config(page_title="Multi-PDF RAG Assistant", layout="wide", page_icon="📚")

st.title("📚 Modular Multi-PDF Multimodal RAG Assistant")
st.write("Upload your research papers to index text in FAISS and visual elements in CLIP, then chat or generate summaries.")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 1. Render Sidebar (returns settings and indexed document list)
use_rerank, indexed_docs = render_sidebar()

# 2. Render Main Chat Interface
render_chat_interface(use_rerank, indexed_docs)
