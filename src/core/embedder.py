import os
import sys
from sentence_transformers import SentenceTransformer
from PIL import Image

# Include root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

import torch

_model = None
_clip_model = None

def get_device():
    """Detects if GPU is available to accelerate embedding generation."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_model():
    """Lazy-loads and caches the SentenceTransformer model for text embeddings (BGE)."""
    global _model
    if _model is None:
        device = get_device()
        print(f"Loading text embedding model: {config.EMBEDDING_MODEL_NAME} on device: {device}...")
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)
    return _model

def get_clip_model():
    """Lazy-loads and caches the CLIP model for multimodal embeddings."""
    global _clip_model
    if _clip_model is None:
        device = get_device()
        print(f"Loading multimodal model: {config.CLIP_MODEL_NAME} on device: {device}...")
        _clip_model = SentenceTransformer(config.CLIP_MODEL_NAME, device=device)
    return _clip_model

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
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings

def embed_image_file(image_path):
    """
    Generates a single normalized float32 vector embedding for an image file using CLIP.
    """
    clip = get_clip_model()
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    img = Image.open(image_path)
    embedding = clip.encode(
        img,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embedding.reshape(1, -1)

def embed_image_batch(image_paths):
    """
    Generates normalized embeddings for a list of image files using CLIP.
    """
    clip = get_clip_model()
    imgs = []
    for path in image_paths:
        if os.path.exists(path):
            imgs.append(Image.open(path))
            
    if not imgs:
        return []
        
    embeddings = clip.encode(
        imgs,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings

def embed_clip_text(query):
    """
    Generates a normalized float32 vector embedding for a text query using CLIP.
    Matches image vectors in CLIP space.
    """
    clip = get_clip_model()
    embedding = clip.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embedding.reshape(1, -1)
