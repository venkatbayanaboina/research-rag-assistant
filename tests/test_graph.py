"""Deterministic transition tests for the LangGraph research graph.

All LLM and retrieval dependencies are stubbed, so these tests exercise the
graph's routing logic without models, API keys, or the FAISS index.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import src.core.agent.graph as graph_module
from src.core.agent.graph import execute_agentic_pipeline, _route_after_sufficiency


def _evidence(n=2, page=1, source="p.pdf"):
    return [{"score": 3.0, "chunk": {"source_file": source, "page": page,
                                     "text": f"chunk {i}", "chunk_id": i}}
            for i in range(n)]


@pytest.fixture()
def stub_llm_free(monkeypatch):
    """Stubs the router and query-expansion LLM calls to be deterministic."""
    monkeypatch.setattr("src.core.generator.router.route_user_intent",
                        lambda prompt, files: {"intent": "STANDARD_CHAT", "target_docs": []})
    monkeypatch.setattr("src.core.generator.orchestrator.generate_similar_queries",
                        lambda prompt: [])


def test_sufficient_evidence_goes_straight_to_answer(monkeypatch, stub_llm_free):
    monkeypatch.setattr("src.core.vector_store.search_store", lambda q, **kw: _evidence(3))
    monkeypatch.setattr("src.core.vector_store.search_image_store", lambda q, **kw: [])
    monkeypatch.setattr("src.core.generator.answering.generate_answer",
                        lambda q, t, i, h: "Grounded factual answer with citation [Source 1].")

    result = execute_agentic_pipeline("what is attention?", ["p.pdf"], use_rerank=True)

    assert result is not None
    assert result["evidence_status"] == "sufficient"
    assert result["refinement_iterations"] == 0
    assert result["verification"]["status"] == "verified"
    assert "[Source 1]" in result["answer"]
    assert result["intent"] == "STANDARD_CHAT"


def test_insufficient_evidence_refines_then_recovers(monkeypatch, stub_llm_free):
    calls = {"n": 0}

    def flaky_search(q, **kw):
        calls["n"] += 1
        return _evidence(2) if calls["n"] > 1 else []  # empty first pass, evidence after refinement

    monkeypatch.setattr("src.core.vector_store.search_store", flaky_search)
    monkeypatch.setattr("src.core.vector_store.search_image_store", lambda q, **kw: [])
    monkeypatch.setattr("src.core.generator.answering.generate_answer",
                        lambda q, t, i, h: "Recovered answer [Source 1].")
    # Re-enable rewrites for this test (fixture stubs them to [])
    monkeypatch.setattr("src.core.generator.orchestrator.generate_similar_queries",
                        lambda prompt: ["refined question about attention"])

    result = execute_agentic_pipeline("obscure question", ["p.pdf"], use_rerank=True)

    assert result is not None
    assert result["refinement_iterations"] == 1
    assert result["evidence_status"] in ("sufficient", "partial")
    assert calls["n"] >= 2


def test_refinement_is_bounded_by_max_iterations(monkeypatch, stub_llm_free):
    monkeypatch.setattr("src.core.vector_store.search_store", lambda q, **kw: [])
    monkeypatch.setattr("src.core.vector_store.search_image_store", lambda q, **kw: [])
    monkeypatch.setattr("src.core.generator.answering.generate_answer",
                        lambda q, t, i, h: "best-effort answer")
    monkeypatch.setattr("src.core.generator.orchestrator.generate_similar_queries",
                        lambda prompt: ["rewrite that still finds nothing"])

    result = execute_agentic_pipeline("impossible question", ["p.pdf"], use_rerank=True)

    assert result is not None
    assert result["refinement_iterations"] == 2  # AGENT_MAX_REFINEMENTS
    assert result["answer"]  # still returns a usable (qualified) response


def test_special_intent_bypasses_retrieval_and_verification(monkeypatch, stub_llm_free):
    monkeypatch.setattr("src.core.generator.router.route_user_intent",
                        lambda prompt, files: {"intent": "COMPARISON",
                                               "target_docs": ["a.pdf", "b.pdf"]})
    monkeypatch.setattr("src.core.vector_store.get_registry",
                        lambda: [{"source_file": "a.pdf", "text": "A text", "page": 1},
                                 {"source_file": "b.pdf", "text": "B text", "page": 1}])
    monkeypatch.setattr("src.core.generator.comparison.generate_comparison",
                        lambda docs_data: "| Dimension | a.pdf | b.pdf |\n|---|---|---|")

    # Retrieval must NOT be called for special intents
    def forbidden(*a, **kw):
        raise AssertionError("retrieval executed for special intent")
    monkeypatch.setattr("src.core.vector_store.search_store", forbidden)
    monkeypatch.setattr("src.core.vector_store.search_image_store", forbidden)

    result = execute_agentic_pipeline("compare a and b", ["a.pdf", "b.pdf"])

    assert result["is_special_intent"] is True
    assert result["verification"]["status"] == "skipped"
    assert "Dimension" in result["answer"]


def test_router_failure_degrades_to_standard_chat(monkeypatch, stub_llm_free):
    def broken_router(prompt, files):
        raise RuntimeError("router LLM unavailable")
    monkeypatch.setattr("src.core.generator.router.route_user_intent", broken_router)
    monkeypatch.setattr("src.core.vector_store.search_store", lambda q, **kw: _evidence(2))
    monkeypatch.setattr("src.core.vector_store.search_image_store", lambda q, **kw: [])
    monkeypatch.setattr("src.core.generator.answering.generate_answer",
                        lambda q, t, i, h: "fallback answer [Source 1].")

    result = execute_agentic_pipeline("some question", ["p.pdf"])

    assert result["intent"] == "STANDARD_CHAT"
    assert any("routing failed" in e.lower() for e in result["errors"])


def test_retrieval_exception_produces_error_state_not_crash(monkeypatch, stub_llm_free):
    def broken_search(q, **kw):
        raise RuntimeError("faiss corrupted")
    monkeypatch.setattr("src.core.vector_store.search_store", broken_search)
    monkeypatch.setattr("src.core.vector_store.search_image_store", lambda q, **kw: [])
    monkeypatch.setattr("src.core.generator.answering.generate_answer",
                        lambda q, t, i, h: "answer")

    result = execute_agentic_pipeline("q", ["p.pdf"])
    # Node-level failure surfaces as insufficient evidence + qualified answer,
    # never as an exception escaping the pipeline.
    assert result is not None
    assert result["answer"]


def test_empty_query_abstains_without_llm_calls(monkeypatch, stub_llm_free):
    result = execute_agentic_pipeline("   ", ["p.pdf"])
    assert result["evidence_status"] == "abstained"
    assert result["model_call_count"] == 0


def test_route_after_sufficiency_transitions():
    assert _route_after_sufficiency({"evidence_status": "sufficient"}) == "generate_answer_node"
    assert _route_after_sufficiency({"evidence_status": "partial"}) == "generate_answer_node"
    assert _route_after_sufficiency({"evidence_status": "insufficient",
                                     "refinement_iteration": 0}) == "refine_query"
    assert _route_after_sufficiency({"evidence_status": "insufficient",
                                     "refinement_iteration": 2}) == "generate_answer_node"
    assert _route_after_sufficiency({"evidence_status": "insufficient",
                                     "refinement_iteration": 0,
                                     "model_call_count": 999}) == "generate_answer_node"


def test_result_shape_is_legacy_superset(monkeypatch, stub_llm_free):
    monkeypatch.setattr("src.core.vector_store.search_store", lambda q, **kw: _evidence(2))
    monkeypatch.setattr("src.core.vector_store.search_image_store", lambda q, **kw: [])
    monkeypatch.setattr("src.core.generator.answering.generate_answer",
                        lambda q, t, i, h: "answer [Source 1].")

    result = execute_agentic_pipeline("q", ["p.pdf"])
    for legacy_key in ("intent", "answer", "image_results", "search_results", "is_special_intent"):
        assert legacy_key in result
    for new_key in ("run_id", "evidence_status", "verification", "model_call_count"):
        assert new_key in result
