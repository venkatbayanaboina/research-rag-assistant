"""Tiny offline smoke checks for CI: the pipeline's core logic with zero
models, zero API calls, zero FAISS. Runs in seconds.
"""

import sys

sys.path.insert(0, ".")


def check(name, fn):
    try:
        fn()
        print(f"[smoke] OK   {name}")
        return True
    except Exception as e:
        print(f"[smoke] FAIL {name}: {e}")
        return False


def t_fusion():
    from src.core.fusion import rrf_fuse
    a = [{"score": 1.0, "chunk": {"source_file": "p", "page": 1, "text": "x"}}]
    fused = rrf_fuse([a, a])
    assert len(fused) == 1 and abs(fused[0]["score"] - 2 / 61) < 1e-9


def t_verification():
    from src.core.verification import verify_citations
    ev = [{"score": 1.0, "chunk": {"source_file": "p.pdf", "page": 1, "text": "t"}}]
    assert verify_citations("A grounded claim [Source 1].", ev)["status"] == "verified"
    assert verify_citations("Bad [Source 9].", ev)["status"] in ("qualified", "failed")


def t_tokenizer():
    from src.core.sparse_retrieval import tokenize
    toks = tokenize("What is the BLEU score of Table 3?")
    assert "bleu" in toks and "the" not in toks


def t_visual_ranking():
    from src.core.visual_ranking import qualify_visual
    assert qualify_visual([], [], k=2) == []


def t_holdout_records_validate():
    """Empty scaffold or validated records — both acceptable states. The runner
    refuses to evaluate until a frozen commit is recorded."""
    from evaluation_suite.pipelines.holdout import load_holdout
    records = load_holdout()  # raises ValueError on any invalid record
    assert isinstance(records, list)


def t_research_router_catalog():
    from src.core.generator.router import ALL_INTENTS, RESEARCH_INTENTS
    assert RESEARCH_INTENTS == {"EVIDENCE_TABLE", "CONTRADICTION", "GAP_EXPLORATION"}
    assert len(ALL_INTENTS) == 7


checks = [("RRF fusion", t_fusion), ("citation verification", t_verification),
          ("BM25 tokenizer", t_tokenizer), ("visual qualification", t_visual_ranking),
          ("holdout records validate", t_holdout_records_validate),
          ("router intent catalog", t_research_router_catalog)]

ok = all(check(n, f) for n, f in checks)
print(f"[smoke] {'ALL PASS' if ok else 'FAILURES'}")
sys.exit(0 if ok else 1)
