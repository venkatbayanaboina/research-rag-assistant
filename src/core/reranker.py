import os
import sys
from sentence_transformers import CrossEncoder

# Include root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

import torch

_reranker = None

def get_reranker():
    """Lazy-loads and caches the CrossEncoder model."""
    global _reranker
    if _reranker is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading reranker model: {config.RERANKER_MODEL_NAME} on device: {device}...")
        _reranker = CrossEncoder(config.RERANKER_MODEL_NAME, device=device)
    return _reranker

def rerank_chunks(query, search_results, enabled=None):
    """
    Reranks search results (lists of {"score": float, "chunk": dict})
    using a joint query-chunk cross-encoder score.
    """
    is_enabled = config.RERANK_ENABLED if enabled is None else enabled
    
    if not search_results:
        return []
        
    if not is_enabled:
        # Just return the top final candidates from the initial list
        return search_results[:config.K_FINAL_CONTEXT]
        
    reranker = get_reranker()
    
    # Prepare query-text pairs for prediction
    pairs = [[query, result["chunk"]["text"]] for result in search_results]
    
    # Calculate scores (cross-encoder returns raw relevance logits)
    print(f"Running Cross-Encoder reranking over {len(search_results)} candidates...")
    scores = reranker.predict(pairs)
    
    # Update scores with cross-encoder outputs
    for idx, score in enumerate(scores):
        search_results[idx]["score"] = float(score)
        
    # Sort results descending by reranker score
    search_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Return top final candidates
    selected_results = search_results[:config.K_FINAL_CONTEXT]
    print(f"Reranking completed. Selected top {len(selected_results)} candidates.")
    return selected_results
