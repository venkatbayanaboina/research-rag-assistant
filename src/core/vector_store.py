import os
import sys
import json
import faiss
import numpy as np
import threading

db_lock = threading.Lock()

# Include root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.core.embedder import embed_batch, embed_text, embed_image_batch, embed_clip_text

def get_registry():
    """Loads the text chunk registry if it exists, otherwise returns an empty list."""
    if os.path.exists(config.CHUNKS_PATH):
        with open(config.CHUNKS_PATH, "r") as f:
            return json.load(f)
    return []

def get_image_registry():
    """Loads the image chunk registry if it exists, otherwise returns an empty list."""
    if os.path.exists(config.IMAGES_REGISTRY_PATH):
        with open(config.IMAGES_REGISTRY_PATH, "r") as f:
            return json.load(f)
    return []

def get_indexed_documents():
    """Returns a list of unique filenames that have been indexed."""
    registry = get_registry()
    return list(sorted(list(set(chunk["source_file"] for chunk in registry))))

def load_text_db():
    """Loads the text FAISS index (1024-dim)."""
    if os.path.exists(config.TEXT_INDEX_PATH):
        print(f"Loading Text FAISS index from {config.TEXT_INDEX_PATH}...")
        return faiss.read_index(config.TEXT_INDEX_PATH)
    print("No existing Text FAISS index found. Creating a new IndexFlatIP (1024-dim)...")
    return faiss.IndexFlatIP(1024)

def load_image_db():
    """Loads the image FAISS index (512-dim)."""
    if os.path.exists(config.IMAGE_INDEX_PATH):
        print(f"Loading Image FAISS index from {config.IMAGE_INDEX_PATH}...")
        return faiss.read_index(config.IMAGE_INDEX_PATH)
    print("No existing Image FAISS index found. Creating a new IndexFlatIP (512-dim)...")
    return faiss.IndexFlatIP(512)

def add_document_to_store(text_chunks, image_chunks):
    """
    Appends text chunks to the BGE text index and visual images to the CLIP index.
    """
    with db_lock:
        filename = None
        if text_chunks:
            filename = text_chunks[0]["source_file"]
        elif image_chunks:
            filename = image_chunks[0]["source_file"]
            
        if not filename:
            return
        
    # 1. Index Text Chunks
    if text_chunks:
        text_registry = get_registry()
        # Avoid duplicate indexing
        if not any(chunk["source_file"] == filename for chunk in text_registry):
            text_index = load_text_db()
            texts = [chunk["text"] for chunk in text_chunks]
            print(f"Generating BGE embeddings for {len(texts)} text chunks...")
            embeddings = embed_batch(texts)
            embeddings = np.array(embeddings, dtype="float32")
            text_index.add(embeddings)
            
            # Map chunk IDs
            start_id = len(text_registry)
            for idx, chunk in enumerate(text_chunks):
                chunk["chunk_id"] = start_id + idx
                text_registry.append(chunk)
                
            with open(config.CHUNKS_PATH, "w") as f:
                json.dump(text_registry, f, indent=2)
            faiss.write_index(text_index, config.TEXT_INDEX_PATH)
            print("Successfully indexed text chunks.")
        else:
            print(f"Text chunks for '{filename}' already indexed.")
            
    # 2. Index Image Chunks (CLIP)
    if image_chunks:
        image_registry = get_image_registry()
        if not any(chunk["source_file"] == filename for chunk in image_registry):
            image_index = load_image_db()
            image_paths = [chunk["image_path"] for chunk in image_chunks]
            print(f"Generating CLIP embeddings for {len(image_paths)} visual chunks...")
            embeddings = embed_image_batch(image_paths)
            if len(embeddings) > 0:
                embeddings = np.array(embeddings, dtype="float32")
                image_index.add(embeddings)
                
                # Map image IDs
                start_id = len(image_registry)
                for idx, chunk in enumerate(image_chunks):
                    chunk["image_id"] = start_id + idx
                    image_registry.append(chunk)
                    
                with open(config.IMAGES_REGISTRY_PATH, "w") as f:
                    json.dump(image_registry, f, indent=2)
                faiss.write_index(image_index, config.IMAGE_INDEX_PATH)
                print("Successfully indexed visual chunks.")
        else:
            print(f"Visual chunks for '{filename}' already indexed.")

def search_store(query, k=None, rerank=None):
    """
    Searches the Text index and applies Cross-Encoder reranking.
    """
    with db_lock:
        index = load_text_db()
        registry = get_registry()
        
        if index.ntotal == 0 or not registry:
            print("Text database is empty.")
            return []
        
    is_rerank = config.RERANK_ENABLED if rerank is None else rerank
    initial_k = config.K_INITIAL_RETRIEVAL if is_rerank else (config.K_FINAL_CONTEXT if k is None else k)
    
    query_vector = embed_text(query, is_query=True)
    scores, indices = index.search(query_vector, k=min(initial_k, index.ntotal))
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and idx < len(registry):
            results.append({
                "score": float(score),
                "chunk": registry[idx]
            })
            
    if is_rerank:
        from src.core.reranker import rerank_chunks
        results = rerank_chunks(query, results, enabled=True)
    else:
        results = results[:k if k is not None else config.K_FINAL_CONTEXT]
        
    return results

def search_image_store(query, k=None):
    """
    Searches the Image index using CLIP text query embedding.
    """
    with db_lock:
        index = load_image_db()
        registry = get_image_registry()
        
        if index.ntotal == 0 or not registry:
            print("Image database is empty.")
            return []
        
    search_k = config.K_IMAGE_RETRIEVAL if k is None else k
    
    # Embed text query in CLIP visual space
    query_vector = embed_clip_text(query)
    
    scores, indices = index.search(query_vector, k=min(search_k, index.ntotal))
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and idx < len(registry):
            score_val = float(score)
            # Only include visual diagrams if they meet the semantic similarity threshold (>= 0.28)
            if score_val >= 0.28:
                results.append({
                    "score": score_val,
                    "chunk": registry[idx]
                })
            
    return results
