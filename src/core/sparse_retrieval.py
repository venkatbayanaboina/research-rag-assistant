"""Sparse (lexical) retrieval over the text chunk registry using BM25.

The BM25 index is built lazily from the same JSON registry that feeds the
dense FAISS store, so the two retrieval modes always see identical corpus
state. The index is cached in-process and invalidated when the registry file
changes (size or mtime), which keeps repeated queries cheap without adding a
persistence layer.
"""

import os
import re
import threading

import config

_bm25_lock = threading.Lock()
_cache = {
    "key": None,       # (chunks_path, mtime, chunk_count)
    "bm25": None,
    "chunks": [],
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Standard + academic filler words that carry no lexical evidence. Digits and
# short technical tokens are always preserved ("fig 3", "bge", "l2").
STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "as", "at", "by",
    "that", "this", "these", "those", "it", "its", "we", "our", "they",
    "their", "he", "she", "his", "her", "you", "your", "i", "me", "my",
    "what", "which", "who", "whom", "how", "when", "where", "why", "does",
    "do", "did", "can", "could", "should", "would", "will", "shall", "may",
    "might", "must", "not", "no", "nor", "but", "if", "then", "than",
    "so", "such", "from", "into", "about", "between", "through", "during",
    "before", "after", "above", "below", "up", "down", "out", "off", "over",
    "under", "again", "further", "once", "here", "there", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "only", "own",
    "same", "too", "very", "just", "also", "give", "gives", "show", "shows",
    "tell", "please", "explain", "describes", "describe", "using", "used",
    "use", "paper", "papers",
})


def tokenize(text):
    """Lowercases and tokenizes text into BM25 terms."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]


def _get_bm25(chunks, cache_key):
    """Builds (or returns the cached) BM25Okapi index over the given chunks."""
    with _bm25_lock:
        if _cache["key"] == cache_key and _cache["bm25"] is not None:
            return _cache["bm25"], _cache["chunks"]
        from rank_bm25 import BM25Okapi
        corpus = [tokenize(c.get("text", "")) for c in chunks]
        bm25 = BM25Okapi(corpus) if corpus else None
        _cache["key"] = cache_key
        _cache["bm25"] = bm25
        _cache["chunks"] = chunks
        return bm25, chunks


def _registry_cache_key(paths):
    path = paths["chunks"]
    try:
        stat = os.stat(path)
        return (path, stat.st_mtime, stat.st_size)
    except OSError:
        return (path, None, None)


def search_store_sparse(query, k=None, session_paths=None):
    """Ranks registry chunks against the query with BM25.

    Returns a list of {"score": float, "chunk": dict} sorted by descending
    BM25 score, truncated to k (defaults to config.BM25_CANDIDATES). Chunks
    with zero matching terms are excluded.
    """
    if k is None:
        k = config.BM25_CANDIDATES

    # Lazy import: keeps tokenizer/ranking usable without faiss/torch loaded
    # (smoke checks and holdout tooling import this module standalone).
    from src.core.vector_store import get_registry, get_paths
    paths = get_paths(session_paths)
    registry = get_registry(session_paths)
    if not registry:
        return []

    # Bind the cache to the file identity so a rebuilt registry invalidates it.
    cache_key = _registry_cache_key(paths)
    bm25, chunks = _get_bm25(registry, cache_key)
    if bm25 is None:
        return []

    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)

    results = []
    for idx, score in ranked[:k * 3]:
        if score <= 0:
            break
        results.append({"score": float(score), "chunk": chunks[idx]})
    return results[:k]
