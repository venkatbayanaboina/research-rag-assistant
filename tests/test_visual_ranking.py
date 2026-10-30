"""Unit tests for the evidence-based visual retrieval logic.

These are pure-function tests (no CLIP, no FAISS): the caption statistics,
IDF-weighted sparse matching, and the qualification rules are exercised with
synthetic registries that mirror the failure modes found in the A/B audit.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.core.visual_ranking import (
    build_caption_stats,
    chunk_key,
    qualify_visual,
    sparse_visual_matches,
)


def _img(source, page, caption, path=None):
    return {"source_file": source, "page": page, "category": "Table",
            "image_path": path or f"/x/{source}_{page}.jpg", "caption": caption}


# A registry mirroring the audit findings: many captions contain the ubiquitous
# tokens "model"/"results" and stray digits; only a few are topically specific.
REGISTRY = [
    _img("a.pdf", 1, "Process AT BT Priority CT P1 0 7 4 14 P2 1 3 2 5"),
    _img("a.pdf", 4, "CMRR = Differential Gain (A)d"),
    _img("a.pdf", 7, "Insert Max-Heap (lower half) Min-Heap (upper half)"),
    _img("a.pdf", 9, "Figure 3: model architecture overview"),
    _img("b.pdf", 2, "Table 4: results on model accuracy"),
    _img("b.pdf", 5, "model results summary table 12"),
    _img("b.pdf", 6, "model results comparison table 13"),
    _img("b.pdf", 7, "model results ablation table 14"),
    _img("b.pdf", 8, "model results table 15"),
    _img("b.pdf", 9, "model results table 16"),
]
DF = build_caption_stats(REGISTRY)


def test_ubiquitous_single_token_does_not_qualify():
    """The core defect: one common token ('model', a stray digit) must not
    surface an image."""
    matches = sparse_visual_matches("explain the model", REGISTRY, DF)
    assert matches == []


def test_stray_digit_alone_does_not_qualify():
    """'Show me Figure 1' returned 2 wrong tables in the old code because the
    digit 1 matched caption text. Now: no explicit caption match, and 'figure'
    is a stopword -> nothing."""
    matches = sparse_visual_matches("Show me Figure 1", REGISTRY, DF)
    assert matches == []


def test_rare_tokens_qualify_with_idf_evidence():
    matches = sparse_visual_matches("CMRR differential gain", REGISTRY, DF)
    keys = [chunk_key(m["chunk"]) for m in matches]
    assert ("a.pdf", 4, "/x/a.pdf_4.jpg") in keys
    # The heap table must not be pulled in by weak token overlap
    assert ("a.pdf", 7, "/x/a.pdf_7.jpg") not in keys


def test_explicit_reference_matches_caption_number():
    matches = sparse_visual_matches("show Figure 3", REGISTRY, DF)
    explicit = [m for m in matches if m["explicit"]]
    assert len(explicit) == 1
    assert chunk_key(explicit[0]["chunk"]) == ("a.pdf", 9, "/x/a.pdf_9.jpg")


def test_explicit_reference_with_roman_numeral():
    registry = [_img("r.pdf", 2, "Table II: latency comparison")]
    df = build_caption_stats(registry)
    matches = sparse_visual_matches("show table ii", registry, df)
    assert matches and matches[0]["explicit"]


def test_agreement_rule_dense_only_is_rejected():
    dense = [{"score": 0.31, "chunk": REGISTRY[0]}]      # dense found it
    sparse = []                                          # no caption evidence
    assert qualify_visual(dense, sparse, k=2) == []


def test_agreement_rule_both_sources_qualify():
    dense = [{"score": 0.31, "chunk": REGISTRY[1]}]
    sparse = [{"score": 0.35, "chunk": REGISTRY[1], "explicit": False, "idf_sum": 3.0}]
    qualified = qualify_visual(dense, sparse, k=2)
    assert len(qualified) == 1
    assert chunk_key(qualified[0]["chunk"]) == chunk_key(REGISTRY[1])


def test_explicit_qualifies_even_without_dense():
    sparse = [{"score": 0.45, "chunk": REGISTRY[3], "explicit": True, "idf_sum": None}]
    qualified = qualify_visual([], sparse, k=2)
    assert len(qualified) == 1
    assert qualified[0]["explicit"] is True


def test_strong_sparse_evidence_qualifies_without_dense():
    # Rare-token match whose IDF sum clears IMAGE_SPARSE_STRONG_IDF
    sparse = [{"score": 0.4, "chunk": REGISTRY[1], "explicit": False,
               "idf_sum": config.IMAGE_SPARSE_STRONG_IDF + 0.5}]
    qualified = qualify_visual([], sparse, k=2)
    assert len(qualified) == 1


def test_weak_sparse_without_agreement_is_rejected():
    sparse = [{"score": 0.3, "chunk": REGISTRY[1], "explicit": False,
               "idf_sum": config.IMAGE_SPARSE_MIN_IDF + 0.1}]  # below STRONG
    assert qualify_visual([], sparse, k=2) == []


def test_k_is_a_cap_not_a_quota():
    dense = [{"score": 0.3, "chunk": REGISTRY[1]}]
    sparse = [{"score": 0.35, "chunk": REGISTRY[1], "explicit": False, "idf_sum": 3.0},
              {"score": 0.3, "chunk": REGISTRY[2], "explicit": False, "idf_sum": 2.8}]
    qualified = qualify_visual(dense, sparse, k=1)
    assert len(qualified) == 1


def test_no_evidence_returns_empty():
    assert qualify_visual([], [], k=2) == []
