import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
RAW_PDF_DIR = os.path.join(STORAGE_DIR, "raw_pdfs")
CHUNKS_PATH = os.path.join(STORAGE_DIR, "chunks.json")

# Multimodal Storage Paths
EXTRACTED_IMAGE_DIR = os.path.join(STORAGE_DIR, "extracted_images")
IMAGES_REGISTRY_PATH = os.path.join(STORAGE_DIR, "images_registry.json")
TEXT_INDEX_PATH = os.path.join(STORAGE_DIR, "text_db.faiss")
IMAGE_INDEX_PATH = os.path.join(STORAGE_DIR, "image_db.faiss")

# Deprecation Alias
INDEX_PATH = TEXT_INDEX_PATH

# Ensure directories exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(RAW_PDF_DIR, exist_ok=True)
os.makedirs(EXTRACTED_IMAGE_DIR, exist_ok=True)

# Model Settings
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
GEMINI_MODEL_NAME = "gemini-1.5-flash"

# CLIP Multimodal Settings
CLIP_MODEL_NAME = "clip-ViT-B-32"
K_IMAGE_RETRIEVAL = 2

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
