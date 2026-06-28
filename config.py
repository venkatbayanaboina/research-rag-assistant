import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
RAW_PDF_DIR = os.path.join(STORAGE_DIR, "raw_pdfs")
CHUNKS_PATH = os.path.join(STORAGE_DIR, "chunks.json")
INDEX_PATH = os.path.join(STORAGE_DIR, "embedding_db.faiss")

# Ensure directories exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(RAW_PDF_DIR, exist_ok=True)

# Model Settings
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
GEMINI_MODEL_NAME = "gemini-3.5-flash"

# Ingestion/Chunking Settings
CHUNK_MAX_CHARACTERS = 2000
CHUNK_NEW_AFTER_CHARACTERS = 1500
CHUNK_COMBINE_UNDER_CHARACTERS = 100
CHUNK_OVERLAP = 200

# Reranker Settings
RERANK_ENABLED = True
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Lightweight and fast on local CPUs
K_INITIAL_RETRIEVAL = 15
K_FINAL_CONTEXT = 5

