import os
import sys
from sentence_transformers import SentenceTransformer

# Include root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

_model = None

def get_model():
    """Lazy-loads and caches the SentenceTransformer model."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model

def embed_text(text, is_query=False):
    """
    Generates a single normalized float32 vector embedding for a given text.
    If is_query is True, prepends the BGE query instruction prefix.
    """
    model = get_model()
    input_text = f"{config.BGE_QUERY_PREFIX}{text}" if is_query else text
    
    embedding = model.encode(
        input_text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embedding.reshape(1, -1)

def embed_batch(texts):
    """
    Generates normalized embeddings for a batch of text snippets.
    Used during chunk ingestion.
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings
