import json, sys, os, time, tempfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.sparse_retrieval import tokenize, search_store_sparse


def test_tokenize_lowercases_and_drops_stopwords_keeps_digits():
    tokens = tokenize("What is the BLEU score of Table 3 in the Transformer paper?")
    assert "bleu" in tokens and "table" in tokens and "3" in tokens
    assert "the" not in tokens and "what" not in tokens and "is" not in tokens


def _write_registry(tmpdir, chunks):
    path = os.path.join(tmpdir, "chunks.json")
    with open(path, "w") as f:
        json.dump(chunks, f)
    return {"chunks": path, "text_index": os.path.join(tmpdir, "t.faiss"),
            "image_index": os.path.join(tmpdir, "i.faiss"),
            "images_registry": os.path.join(tmpdir, "ir.json"),
            "raw_pdfs": tmpdir, "extracted_images": tmpdir}


def _chunk(source, page, text, cid=0):
    return {"chunk_id": cid, "source_file": source, "text": text, "page": page,
            "section_title": "S", "images": [], "tables": []}


def test_bm25_ranks_exact_lexical_match_first(tmp_path):
    chunks = [
        _chunk("a.pdf", 1, "The transformer uses stacked self-attention and feed-forward layers.", 0),
        _chunk("a.pdf", 2, "We evaluate on the WMT 2014 English-German translation task with BLEU scores.", 1),
        _chunk("b.pdf", 1, "Table 3 reports BLEU scores for the WMT 2014 English-German task: 28.4.", 2),
        _chunk("b.pdf", 2, "Convolutional networks apply filters to sentence matrices for classification.", 3),
    ]
    paths = _write_registry(str(tmp_path), chunks)

    results = search_store_sparse("Table 3 BLEU WMT 2014 English-German", k=2, session_paths=paths)

    assert results, "BM25 returned no results for an exact lexical match"
    assert results[0]["chunk"]["chunk_id"] == 2
    assert results[0]["score"] > 0


def test_bm25_excludes_zero_score_chunks(tmp_path):
    chunks = [_chunk("a.pdf", 1, "Completely unrelated text about gardening.", 0)]
    paths = _write_registry(str(tmp_path), chunks)
    results = search_store_sparse("quantum entanglement", k=5, session_paths=paths)
    assert results == []


def test_bm25_empty_registry_returns_empty(tmp_path):
    paths = _write_registry(str(tmp_path), [])
    assert search_store_sparse("anything", k=5, session_paths=paths) == []


def test_bm25_cache_invalidates_when_registry_changes(tmp_path):
    # Three-chunk corpus keeps IDF positive (BM25Okapi idf = log((N-n+0.5)/(n+0.5))
    # is <= 0 whenever a term matches >= half the corpus, e.g. 1-of-2 chunks).
    chunks_v1 = [
        _chunk("a.pdf", 1, "target passage about reciprocity norms.", 0),
        _chunk("a.pdf", 2, "Unrelated filler content about gardening tools entirely.", 1),
        _chunk("a.pdf", 3, "More filler text about weather patterns and seasons.", 2),
    ]
    paths = _write_registry(str(tmp_path), chunks_v1)
    r1 = search_store_sparse("reciprocity norms", k=3, session_paths=paths)
    assert len(r1) == 1

    chunks_v2 = chunks_v1 + [
        _chunk("a.pdf", 4, "reciprocity norms revisited with more detail here.", 3),
        # Keep 'reciprocity' below half the corpus: 2 matches of 5 chunks.
        _chunk("a.pdf", 5, "Additional background material on distributed systems theory.", 4),
    ]
    with open(paths["chunks"], "w") as f:
        json.dump(chunks_v2, f)
    # Ensure mtime granularity cannot mask the change
    os.utime(paths["chunks"], (time.time() + 2, time.time() + 2))

    r2 = search_store_sparse("reciprocity norms", k=3, session_paths=paths)
    assert len(r2) == 2
