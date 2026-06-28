import os
import sys
import json
import faiss
import numpy as np

# Include root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.core.embedder import embed_batch, embed_text

def get_registry():
    """Loads the chunk registry if it exists, otherwise returns an empty list."""
    if os.path.exists(config.CHUNKS_PATH):
        with open(config.CHUNKS_PATH, "r") as f:
            return json.load(f)
    return []

def get_indexed_documents():
    """Returns a list of unique filenames that have been indexed."""
    registry = get_registry()
    return list(sorted(list(set(chunk["source_file"] for chunk in registry))))

def load_vector_db():
    """Loads the FAISS index. If it does not exist, returns a new IndexFlatIP (1024)."""
    if os.path.exists(config.INDEX_PATH):
        print(f"Loading FAISS index from {config.INDEX_PATH}...")
        return faiss.read_index(config.INDEX_PATH)
    print("No existing FAISS index found. Creating a new IndexFlatIP (1024-dim)...")
    return faiss.IndexFlatIP(1024)

def add_document_to_store(processed_chunks):
    """
    Adds a set of processed chunks (from a single document) to the vector store.
    Updates the FAISS index and the chunks.json registry.
    """
    if not processed_chunks:
        return
        
    filename = processed_chunks[0]["source_file"]
    registry = get_registry()
    
    # Check if this document is already in the registry to prevent duplicates
    if any(chunk["source_file"] == filename for chunk in registry):
        print(f"Document '{filename}' is already indexed. Skipping.")
        return
        
    index = load_vector_db()
    
    # Extract raw text from chunks and generate embeddings
    texts = [chunk["text"] for chunk in processed_chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = embed_batch(texts)
    
    # Reshape and add to FAISS index
    embeddings = np.array(embeddings, dtype="float32")
    index.add(embeddings)
    
    # Assign chunk_ids and append to main registry
    start_id = len(registry)
    for idx, chunk in enumerate(processed_chunks):
        chunk["chunk_id"] = start_id + idx
        registry.append(chunk)
        
    # Write updated chunks registry to disk
    with open(config.CHUNKS_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"Updated chunk registry at {config.CHUNKS_PATH}")
    
    # Save the updated FAISS index to disk
    faiss.write_index(index, config.INDEX_PATH)
    print(f"Saved updated FAISS index to {config.INDEX_PATH}")

def search_store(query, k=None, rerank=None):
    """
    Searches the FAISS index for matching chunks, applying reranking if enabled.
    Returns a list of dicts: {"score": float, "chunk": dict}
    """
    index = load_vector_db()
    registry = get_registry()
    
    if index.ntotal == 0 or not registry:
        print("Vector database is empty. No search can be performed.")
        return []
        
    # Check configurations
    is_rerank = config.RERANK_ENABLED if rerank is None else rerank
    
    # Define retrieval depth
    initial_k = config.K_INITIAL_RETRIEVAL if is_rerank else (config.K_FINAL_CONTEXT if k is None else k)
    
    # Embed user query
    query_vector = embed_text(query, is_query=True)
    
    # Search FAISS
    scores, indices = index.search(query_vector, k=min(initial_k, index.ntotal))
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        # Safely map to chunk (FAISS index -1 represents no match)
        if idx != -1 and idx < len(registry):
            results.append({
                "score": float(score),
                "chunk": registry[idx]
            })
            
    # Apply Cross-Encoder reranking
    if is_rerank:
        from src.core.reranker import rerank_chunks
        results = rerank_chunks(query, results, enabled=True)
    else:
        results = results[:k if k is not None else config.K_FINAL_CONTEXT]
        
    return results
