import os
import sys
import json
import faiss
import numpy as np
import threading

# Reentrant so compound write paths (e.g. add_document_to_store purging the
# same document before appending) can nest acquisitions on one thread.
db_lock = threading.RLock()

# Include root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.core.embedder import embed_batch, embed_text, embed_image_batch, embed_clip_text

def get_paths(session_paths=None):
    """
    Returns the active database paths. If session_paths is explicitly passed,
    uses it. If running inside a Streamlit user session, returns session-isolated
    temporary paths, otherwise falls back to config.
    """
    if session_paths:
        return session_paths
    try:
        import streamlit as st
        if st.session_state and "session_id" in st.session_state:
            return {
                "chunks": st.session_state.CHUNKS_PATH,
                "text_index": st.session_state.TEXT_INDEX_PATH,
                "image_index": st.session_state.IMAGE_INDEX_PATH,
                "images_registry": st.session_state.IMAGES_REGISTRY_PATH,
                "raw_pdfs": st.session_state.RAW_PDF_DIR,
                "extracted_images": st.session_state.EXTRACTED_IMAGE_DIR
            }
    except:
        pass
    
    return {
        "chunks": config.CHUNKS_PATH,
        "text_index": config.TEXT_INDEX_PATH,
        "image_index": config.IMAGE_INDEX_PATH,
        "images_registry": config.IMAGES_REGISTRY_PATH,
        "raw_pdfs": config.RAW_PDF_DIR,
        "extracted_images": config.EXTRACTED_IMAGE_DIR
    }

def get_registry(session_paths=None):
    """Loads the text chunk registry if it exists, otherwise returns an empty list."""
    paths = get_paths(session_paths)
    if os.path.exists(paths["chunks"]):
        with open(paths["chunks"], "r") as f:
            return json.load(f)
    return []

def get_image_registry(session_paths=None):
    """Loads the image chunk registry if it exists, otherwise returns an empty list."""
    paths = get_paths(session_paths)
    if os.path.exists(paths["images_registry"]):
        with open(paths["images_registry"], "r") as f:
            return json.load(f)
    return []

def get_indexed_documents(session_paths=None):
    """Returns a list of unique filenames that have been indexed."""
    registry = get_registry(session_paths)
    return list(sorted(list(set(chunk["source_file"] for chunk in registry))))

def load_text_db(session_paths=None):
    """Loads the text FAISS index (1024-dim)."""
    paths = get_paths(session_paths)
    if os.path.exists(paths["text_index"]):
        print(f"Loading Text FAISS index from {paths['text_index']}...")
        return faiss.read_index(paths["text_index"])
    print("No existing Text FAISS index found. Creating a new IndexFlatIP (1024-dim)...")
    return faiss.IndexFlatIP(1024)

def load_image_db(session_paths=None):
    """Loads the image FAISS index (512-dim)."""
    paths = get_paths(session_paths)
    if os.path.exists(paths["image_index"]):
        print(f"Loading Image FAISS index from {paths['image_index']}...")
        return faiss.read_index(paths["image_index"])
    print("No existing Image FAISS index found. Creating a new IndexFlatIP (512-dim)...")
    return faiss.IndexFlatIP(512)

def add_document_to_store(text_chunks, image_chunks, session_paths=None):
    """
    Appends text chunks to the BGE text index and visual images to the CLIP index.

    The whole read-embed-write sequence runs under db_lock: FAISS row order must
    stay aligned with the registry append order, so concurrent writers (or a
    concurrent delete-rebuild) interleaving with these steps would corrupt IDs.
    """
    filename = None
    if text_chunks:
        filename = text_chunks[0]["source_file"]
    elif image_chunks:
        filename = image_chunks[0]["source_file"]

    if not filename:
        return

    with db_lock:
        _add_document_to_store_locked(text_chunks, image_chunks, filename, session_paths)


def _add_document_to_store_locked(text_chunks, image_chunks, filename, session_paths):
    # Auto-delete existing records of the same document to support clean re-indexing/overwriting
    delete_document_from_store(filename, session_paths=session_paths)
    
    # 1. Index Text Chunks
    if text_chunks:
        text_registry = get_registry(session_paths)
        # Avoid duplicate indexing
        if not any(chunk["source_file"] == filename for chunk in text_registry):
            # Load registry and indexes
            text_index = load_text_db(session_paths)
            texts = [chunk["text"] for chunk in text_chunks]
            print(f"Generating BGE embeddings for {len(texts)} text chunks...")
            
            import time
            from src.core.utils.profiler import log_timing
            
            start_time = time.time()
            
            embeddings = embed_batch(texts)
            duration = time.time() - start_time
            log_timing(
                step_name="text_embedding_generation",
                duration_seconds=duration,
                metadata={
                    "file_name": filename,
                    "chunk_count": len(texts)
                }
            )
            embeddings = np.array(embeddings, dtype="float32")
            text_index.add(embeddings)
            
            # Map chunk IDs
            start_id = len(text_registry)
            for idx, chunk in enumerate(text_chunks):
                chunk["chunk_id"] = start_id + idx
                text_registry.append(chunk)
                
            paths = get_paths(session_paths)
            with open(paths["chunks"], "w") as f:
                json.dump(text_registry, f, indent=2)
            faiss.write_index(text_index, paths["text_index"])
            print("Successfully indexed text chunks.")
        else:
            print(f"Text chunks for '{filename}' already indexed.")
            
    # 2. Index Image Chunks (CLIP)
    if image_chunks:
        image_registry = get_image_registry(session_paths)
        if not any(chunk["source_file"] == filename for chunk in image_registry):
            image_index = load_image_db(session_paths)
            image_paths = [chunk["image_path"] for chunk in image_chunks]
            print(f"Generating CLIP embeddings for {len(image_paths)} visual chunks...")
            
            import time
            from src.core.utils.profiler import log_timing
            
            start_time = time.time()
            embeddings = embed_image_batch(image_paths)
            duration = time.time() - start_time
            log_timing(
                step_name="visual_embedding_generation",
                duration_seconds=duration,
                metadata={
                    "file_name": filename,
                    "image_count": len(image_paths)
                }
            )
            if len(embeddings) > 0:
                embeddings = np.array(embeddings, dtype="float32")
                image_index.add(embeddings)
                
                # Map image IDs
                start_id = len(image_registry)
                for idx, chunk in enumerate(image_chunks):
                    chunk["image_id"] = start_id + idx
                    image_registry.append(chunk)
                    
                paths = get_paths(session_paths)
                with open(paths["images_registry"], "w") as f:
                    json.dump(image_registry, f, indent=2)
                faiss.write_index(image_index, paths["image_index"])
                print("Successfully indexed visual chunks.")
        else:
            print(f"Visual chunks for '{filename}' already indexed.")

def search_store(query, k=None, rerank=None, mode=None, query_vector=None):
    """
    Searches the Text index and applies Cross-Encoder reranking.

    mode="dense"  — legacy dense-only BGE/FAISS retrieval (unchanged behavior).
    mode="hybrid" — BM25 sparse ranking fused with the dense ranking via RRF
                    before reranking. Defaults to config.HYBRID_RETRIEVAL_ENABLED.
    query_vector  — optional pre-computed BGE query embedding (from
                    embedder.embed_queries) to skip per-query encoding in
                    multi-query paths.
    """
    import time
    from src.core.utils.profiler import log_timing
    start_time = time.time()

    with db_lock:
        index = load_text_db()
        registry = get_registry()

        if index.ntotal == 0 or not registry:
            print("Text database is empty.")
            return []

    if mode is None:
        mode = "hybrid" if config.HYBRID_RETRIEVAL_ENABLED else "dense"

    is_rerank = config.RERANK_ENABLED if rerank is None else rerank
    initial_k = config.K_INITIAL_RETRIEVAL if is_rerank else (config.K_FINAL_CONTEXT if k is None else k)

    if query_vector is None:
        query_vector = embed_text(query, is_query=True)
    scores, indices = index.search(np.atleast_2d(np.asarray(query_vector, dtype="float32")),
                                   k=min(initial_k, index.ntotal))

    results = []
    seen_texts = set()
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and idx < len(registry):
            chunk = registry[idx]
            # Clean and normalize the text for comparison
            normalized_text = " ".join(chunk["text"].lower().strip().split())
            if normalized_text not in seen_texts:
                seen_texts.add(normalized_text)
                results.append({
                    "score": float(score),
                    "chunk": chunk
                })

    dense_results = results

    if mode == "hybrid":
        from src.core.sparse_retrieval import search_store_sparse
        from src.core.fusion import rrf_fuse
        sparse_results = search_store_sparse(query, k=config.BM25_CANDIDATES)
        results = rrf_fuse([dense_results, sparse_results], top_k=initial_k)
    else:
        sparse_results = []

    # Measure initial FAISS text search duration
    duration = time.time() - start_time
    log_timing(
        step_name="text_retrieval_faiss",
        duration_seconds=duration,
        metadata={
            "query": query[:100],
            "initial_candidates": len(results),
            "mode": mode,
            "sparse_candidates": len(sparse_results)
        }
    )

    if is_rerank:
        from src.core.reranker import rerank_chunks
        results = rerank_chunks(query, results, enabled=True)
    else:
        results = results[:k if k is not None else config.K_FINAL_CONTEXT]

    return results

def search_image_store(query, k=None, session_paths=None):
    """
    Searches the Image index using Hybrid Visual Retrieval with calibrated
    qualification:

      1. Dense ranking: CLIP text-to-image similarity, adaptively filtered
         (relative margin from the best hit + absolute noise floor).
      2. Sparse ranking: IDF-weighted caption matching; explicit
         "figure/table N" references always qualify.
      3. Fusion + evidence rules (src/core/visual_ranking.py): an image is
         returned only on explicit reference, dense∩sparse agreement, or
         strong caption evidence. Returns [] when nothing genuinely matches —
         k is a cap, never a quota.
    """
    import time
    from src.core.utils.profiler import log_timing
    start_time = time.time()

    with db_lock:
        index = load_image_db(session_paths)
        registry = get_image_registry(session_paths)

        if index.ntotal == 0 or not registry:
            print("Image database is empty.")
            return []

    max_images = config.K_IMAGE_RETRIEVAL if k is None else k

    # 1. Dense ranking with per-query adaptive filtering
    query_vector = embed_clip_text(query)
    scores, indices = index.search(query_vector, k=min(max_images * 5, index.ntotal))

    dense_ranked = []
    top_dense = None
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(registry):
            continue
        s = float(score)
        if top_dense is None:
            top_dense = s
        if s < config.IMAGE_DENSE_ABS_FLOOR:
            break  # results arrive in descending order
        if s < top_dense - config.IMAGE_DENSE_RELATIVE_MARGIN:
            break
        dense_ranked.append({"score": s, "chunk": registry[idx]})

    # 2. Sparse ranking (IDF-weighted captions + explicit references)
    from src.core.visual_ranking import build_caption_stats, sparse_visual_matches, qualify_visual
    caption_df = build_caption_stats(registry)
    sparse_ranked = sparse_visual_matches(query, registry, caption_df)

    # 3. Fusion + evidence-based qualification
    qualified = qualify_visual(dense_ranked, sparse_ranked, max_images)

    duration = time.time() - start_time
    log_timing(
        step_name="visual_retrieval_clip",
        duration_seconds=duration,
        metadata={
            "query": query[:100],
            "dense_candidates": len(dense_ranked),
            "sparse_candidates": len(sparse_ranked),
            "final_results_count": len(qualified)
        }
    )
    return qualified

def delete_document_from_store(filename, session_paths=None):
    """
    Purges a document's text and visual chunks from both registries
    and reconstructs the FAISS indexes from the remaining vectors (API-free).
    """
    with db_lock:
        registry = get_registry(session_paths)
        image_registry = get_image_registry(session_paths)
        
        # 1. Identify indices to keep
        keep_text_indices = [idx for idx, c in enumerate(registry) if c["source_file"] != filename]
        keep_image_indices = [idx for idx, c in enumerate(image_registry) if c["source_file"] != filename]
        
        # If nothing needs to be deleted, bypass reconstruction to save time
        if len(keep_text_indices) == len(registry) and len(keep_image_indices) == len(image_registry):
            return
            
        new_registry = [registry[idx] for idx in keep_text_indices]
        new_image_registry = [image_registry[idx] for idx in keep_image_indices]
        
        # Re-map IDs to match new array positions
        for idx, chunk in enumerate(new_registry):
            chunk["chunk_id"] = idx
        for idx, chunk in enumerate(new_image_registry):
            chunk["image_id"] = idx
            
        # 2. Reconstruct Text Index
        old_text_index = load_text_db(session_paths)
        new_text_index = faiss.IndexFlatIP(1024)
        if keep_text_indices and old_text_index.ntotal > 0:
            vectors = []
            for idx in keep_text_indices:
                if idx < old_text_index.ntotal:
                    vectors.append(old_text_index.reconstruct(idx))
            if vectors:
                new_text_index.add(np.array(vectors, dtype="float32"))
                
        # 3. Reconstruct Image Index
        old_image_index = load_image_db(session_paths)
        new_image_index = faiss.IndexFlatIP(512)
        if keep_image_indices and old_image_index.ntotal > 0:
            img_vectors = []
            for idx in keep_image_indices:
                if idx < old_image_index.ntotal:
                    img_vectors.append(old_image_index.reconstruct(idx))
            if img_vectors:
                new_image_index.add(np.array(img_vectors, dtype="float32"))
                
        # 4. Save updated registries and FAISS indexes
        paths = get_paths(session_paths)
        with open(paths["chunks"], "w") as f:
            json.dump(new_registry, f, indent=2)
            
        with open(paths["images_registry"], "w") as f:
            json.dump(new_image_registry, f, indent=2)
            
        faiss.write_index(new_text_index, paths["text_index"])
        faiss.write_index(new_image_index, paths["image_index"])
        
        print(f"Successfully purged '{filename}' from database stores.")

def clear_all_documents_from_store(session_paths=None):
    """
    Wipes all indexed documents, text registry, CLIP image registry,
    and FAISS indexes for the current database session.
    """
    with db_lock:
        paths = get_paths(session_paths)
        # Delete database index files
        for key in ["chunks", "images_registry", "text_index", "image_index"]:
            path = paths.get(key)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
                    
        # Clear cropped images directory
        img_dir = paths.get("extracted_images")
        if img_dir and os.path.exists(img_dir):
            import shutil
            try:
                shutil.rmtree(img_dir)
                os.makedirs(img_dir, exist_ok=True)
            except:
                pass

