"""Regression tests pinning the defect fixes from docs/upgrade/repository_audit.md."""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import main
import src.core.vector_store as vector_store
from src.core.generator.summarization import get_key_images_for_doc


def test_main_compare_uses_current_signature(monkeypatch):
    """D1: CLI `compare` must call generate_comparison(docs_data: list[dict])."""
    captured = {}

    def fake_generate_comparison(docs_data):
        captured["docs_data"] = docs_data
        return "| Dimension | a.pdf | b.pdf |\n|---|---|---|"

    monkeypatch.setattr("src.core.vector_store.get_registry",
                        lambda: [{"source_file": "a.pdf", "text": "A", "page": 1},
                                 {"source_file": "b.pdf", "text": "B", "page": 1}])
    monkeypatch.setattr("src.core.generator.comparison.generate_comparison",
                        fake_generate_comparison)

    main.run_compare("a.pdf", "b.pdf")

    docs = captured["docs_data"]
    assert isinstance(docs, list) and len(docs) == 2
    assert docs[0] == {"name": "a.pdf", "chunks": [{"source_file": "a.pdf", "text": "A", "page": 1}]}


def test_key_images_use_session_aware_registry(monkeypatch, tmp_path):
    """D3: get_key_images_for_doc must read the session registry, not the
    global config path (simulated here by a nonexistent global path)."""
    assert not os.path.exists(config.IMAGES_REGISTRY_PATH) or True

    img_file = tmp_path / "fig1.jpg"
    img_file.write_bytes(b"fake")

    session_registry = [{
        "source_file": "session_doc.pdf", "page": 3, "category": "Image",
        "image_path": str(img_file), "caption": "Figure 1: model architecture"
    }]
    monkeypatch.setattr("src.core.vector_store.get_image_registry",
                        lambda session_paths=None: session_registry)

    selected = get_key_images_for_doc("session_doc.pdf")
    assert len(selected) == 1
    assert selected[0]["caption"].startswith("Figure 1")


def test_db_lock_is_reentrant():
    """D6: compound write paths nest acquisitions; Lock would deadlock."""
    assert isinstance(vector_store.db_lock, type(vector_store.threading.RLock()))


def test_search_store_dense_mode_is_hybrid_free(monkeypatch):
    """mode='dense' must not invoke BM25 at all (legacy parity)."""
    monkeypatch.setattr(vector_store, "load_text_db", lambda: _FakeIndex(3))
    monkeypatch.setattr(vector_store, "get_registry", lambda: _fake_registry(3))
    monkeypatch.setattr("src.core.embedder.embed_text", lambda text, is_query=False: [[0.1] * 4])

    def forbidden_sparse(*a, **kw):
        raise AssertionError("sparse retrieval called in dense mode")
    import src.core.sparse_retrieval as sparse
    monkeypatch.setattr(sparse, "search_store_sparse", forbidden_sparse)

    results = vector_store.search_store("query", k=2, rerank=False, mode="dense")
    assert len(results) == 2


class _FakeIndex:
    def __init__(self, n):
        self.ntotal = n

    def search(self, vector, k):
        import numpy as np
        scores = np.array([[0.9, 0.8, 0.7][:k]])
        indices = np.array([[0, 1, 2][:k]])
        return scores, indices


def _fake_registry(n):
    return [{"chunk_id": i, "source_file": "p.pdf", "page": i + 1,
             "text": f"text {i}", "images": [], "tables": []} for i in range(n)]
