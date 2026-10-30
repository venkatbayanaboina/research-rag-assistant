import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.fusion import rrf_fuse


def _chunk(source, page, text):
    return {"source_file": source, "page": page, "text": text}


def test_rrf_fuse_merges_and_scores():
    a = [{"score": 0.9, "chunk": _chunk("p1.pdf", 1, "alpha")}]
    b = [{"score": 2.0, "chunk": _chunk("p1.pdf", 1, "alpha")},
         {"score": 1.0, "chunk": _chunk("p1.pdf", 2, "beta")}]

    fused = rrf_fuse([a, b])

    # alpha appears at rank 1 in both rankings: 1/(60+1) + 1/(60+1)
    alpha = fused[0]
    assert alpha["chunk"]["text"] == "alpha"
    assert abs(alpha["score"] - 2.0 / 61) < 1e-9
    # beta appears only in ranking b at rank 2
    beta = fused[1]
    assert beta["chunk"]["text"] == "beta"
    assert abs(beta["score"] - 1.0 / 62) < 1e-9


def test_rrf_fuse_orders_by_contribution_not_raw_scores():
    # Raw similarity scores must be ignored; only rank positions matter.
    # The tiny-score chunk holds rank 1, the huge-score chunk rank 2.
    a = [{"score": 0.1, "chunk": _chunk("p.pdf", 1, "rank_one_tiny_score")},
         {"score": 999.0, "chunk": _chunk("p.pdf", 2, "rank_two_huge_score")}]
    fused = rrf_fuse([a])
    assert fused[0]["chunk"]["text"] == "rank_one_tiny_score"


def test_rrf_fuse_dedupes_identical_chunks_across_rankings():
    a = [{"score": 1.0, "chunk": _chunk("p.pdf", 1, "dup")}]
    b = [{"score": 0.5, "chunk": _chunk("p.pdf", 1, "dup")}]
    c = [{"score": 0.5, "chunk": _chunk("p.pdf", 1, "dup")}]
    fused = rrf_fuse([a, b, c])
    assert len(fused) == 1
    assert abs(fused[0]["score"] - 3.0 / 61) < 1e-9


def test_rrf_fuse_empty_and_topk():
    assert rrf_fuse([]) == []
    a = [{"score": 1.0, "chunk": _chunk("p.pdf", i, f"t{i}")} for i in range(5)]
    assert len(rrf_fuse([a], top_k=2)) == 2


def test_rrf_fuse_handles_chunk_identity_via_image_path():
    img_a = {"source_file": "p.pdf", "page": 1, "image_path": "/x/fig1.jpg"}
    img_b = {"source_file": "p.pdf", "page": 1, "image_path": "/x/fig1.jpg"}
    fused = rrf_fuse([[{"score": 1.0, "chunk": img_a}], [{"score": 1.0, "chunk": img_b}]])
    assert len(fused) == 1
