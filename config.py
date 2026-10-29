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
GEMINI_MODEL_NAME = "models/gemini-2.5-flash"

# CLIP Multimodal Settings
CLIP_MODEL_NAME = "clip-ViT-B-32"
K_IMAGE_RETRIEVAL = 2           # MAX images attached per answer (a cap, not a quota —
                                # fewer or zero are returned when nothing qualifies)
IMAGE_DENSE_ABS_FLOOR = 0.20    # CLIP similarity noise floor (absolute)
IMAGE_DENSE_RELATIVE_MARGIN = 0.06  # keep dense hits within this distance of the best hit
IMAGE_SPARSE_MIN_IDF = 1.2      # caption-token evidence (IDF sum) needed for a sparse match
IMAGE_SPARSE_STRONG_IDF = 2.5   # caption evidence strong enough to qualify without dense agreement

# Ingestion/Chunking Settings
DEFAULT_PARSE_STRATEGY = "hi_res"  # "fast" stays available via CLI for text-only dumps
CHUNK_MAX_CHARACTERS = 2000
CHUNK_NEW_AFTER_CHARACTERS = 1500
CHUNK_COMBINE_UNDER_CHARACTERS = 100
CHUNK_OVERLAP = 200

# Reranker Settings
RERANK_ENABLED = True
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Lightweight and fast on local CPUs
K_INITIAL_RETRIEVAL = 15
K_FINAL_CONTEXT = 5

# Hybrid Text Retrieval Settings (BM25 sparse + BGE dense fused via RRF)
HYBRID_RETRIEVAL_ENABLED = True
RRF_K_CONSTANT = 60              # Standard RRF dampening constant
BM25_CANDIDATES = 15             # Sparse candidates per query before fusion

# Agentic Orchestration Settings (LangGraph)
AGENTIC_PIPELINE_ENABLED = True  # False restores the legacy pipeline verbatim
AGENT_MAX_REFINEMENTS = 2        # Bounded retrieval-refinement iterations
AGENT_MODEL_CALL_BUDGET = 12     # Hard cap on LLM calls per query
AGENT_DEADLINE_SECONDS = 120     # Wall-clock deadline per query
EVIDENCE_MIN_CHUNKS = 2          # Sufficiency: require at least this many chunks
EVIDENCE_MIN_TOP_SCORE = 0.5     # Sufficiency: cross-encoder/RRF score floor (0..1 heuristic)
EVIDENCE_TARGET_COVERAGE = 0.34  # Sufficiency: min fraction of target docs present in evidence

# Claim-Level Citation Verification
VERIFY_CLAIMS_WITH_LLM = False   # Layer 2 semantic verification (extra LLM call, opt-in)

# Judge model selection (item 7, hardening). Default: judge with the gate
# rotation head. Set JUDGE_MODEL to any OpenRouter slug to judge with a
# different model family than the generator (results are written to a
# separate file and reported separately).
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "").strip() or None
