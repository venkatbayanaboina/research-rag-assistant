"""Tests for the Phase 5 agentic research actions (ADR-008).

The LLM is stubbed at the structured-call boundary (_llm_json); everything
else — provenance validation, the no-guesses rule, classification schemas,
action logging, and graph routing for the research intents — is tested for
real.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import src.core.research_actions as ra
from src.core.research_actions import (
    NOT_STATED,
    _valid_pages,
    run_research_actions,
)


def _registry():
    return [
        {"chunk_id": 0, "source_file": "a.pdf", "page": 1, "text": "We propose X on dataset D."},
        {"chunk_id": 1, "source_file": "a.pdf", "page": 5, "text": "Results: 91% accuracy."},
        {"chunk_id": 2, "source_file": "a.pdf", "page": 7, "text": "Limitation: small dataset."},
        {"chunk_id": 3, "source_file": "b.pdf", "page": 2, "text": "We propose Y on dataset E."},
        {"chunk_id": 4, "source_file": "b.pdf", "page": 4, "text": "Results: 88% accuracy."},
    ]


def _stub(monkeypatch, responses):
    """Stubs the structured LLM calls; pops one canned response per call."""
    it = iter(responses)
    monkeypatch.setattr(ra, "_llm_json", lambda prompt, system_instruction=None: next(it))


# ---------------------------------------------------------- provenance

def test_valid_pages_drops_fabricated_pages():
    allowed = {1, 5, 7}
    assert _valid_pages([1, 5, 99, "7", "x", 1], allowed) == [1, 5, 7]


def test_evidence_table_never_guesses_missing_fields(monkeypatch):
    _stub(monkeypatch, [{
        "research_question": None, "method": "Approach X", "dataset": "Dataset D",
        "metric": "accuracy", "result": "91%", "limitations": "small dataset",
        "evidence_pages": [5, 42],   # 42 does not exist in a.pdf -> must be dropped
    }])
    out = run_research_actions("EVIDENCE_TABLE", "", ["a.pdf"], _registry())
    assert "| Paper |" in out["markdown"] and "Approach X" in out["markdown"]
    assert NOT_STATED in out["markdown"]          # null field rendered honestly
    row = out["data"][0]
    assert row["evidence_pages"] == [5]           # fabricated page 42 dropped
    assert any(a["tool"] == "evidence_table.extract" for a in out["actions"])


def test_evidence_table_budget_stops_extraction(monkeypatch):
    def counting_stub(prompt, system_instruction=None):
        return {"research_question": None, "method": "m", "dataset": None, "metric": None,
                "result": None, "limitations": None, "evidence_pages": []}
    monkeypatch.setattr(ra, "_llm_json", counting_stub)
    # Emulate an exhausted agent budget: routing already spent the last call.
    monkeypatch.setattr(ra.config, "AGENT_MODEL_CALL_BUDGET", 1)

    out = run_research_actions("EVIDENCE_TABLE", "", ["a.pdf", "b.pdf"], _registry())
    assert any(a.get("skipped") for a in out["actions"])
    assert "No evidence" in out["markdown"]


# ---------------------------------------------------------- contradiction

def test_contradiction_classifies_and_uses_valid_classes(monkeypatch):
    _stub(monkeypatch, [
        {"claim": "accuracy 91% with X on D", "page": 5, "conditions": "Dataset D"},
        {"claim": "accuracy 88% with Y on E", "page": 4, "conditions": "Dataset E"},
        {"contradictions": [
            {"papers": ["a.pdf", "b.pdf"], "classification": "different_datasets",
             "explanation": "Numbers differ because datasets differ, not because findings conflict."},
        ]},
    ])
    out = run_research_actions("CONTRADICTION", "classification accuracy",
                               ["a.pdf", "b.pdf"], _registry())
    assert "`different_datasets`" in out["markdown"]
    assert "not as genuine contradictions" in out["markdown"]
    verdicts = out["data"]["verdicts"]
    assert verdicts[0]["classification"] in {
        "genuine_disagreement", "different_datasets", "different_evaluation_settings",
        "different_definitions", "not_a_contradiction"}


def test_contradiction_needs_two_claims(monkeypatch):
    _stub(monkeypatch, [
        {"claim": "only paper a addresses the topic", "page": 5, "conditions": None},
    ])
    out = run_research_actions("CONTRADICTION", "topic", ["a.pdf", "b.pdf"], _registry())
    assert "at least two claims" in out["markdown"]
    assert any(a.get("skipped") for a in out["actions"])


# ---------------------------------------------------------- gap exploration

def test_gap_report_labels_evidence_patterns_and_hypotheses(monkeypatch):
    _stub(monkeypatch, [
        {"limitations": [{"text": "small dataset", "page": 7},
                         {"text": "future: multilingual", "page": 44}]},   # 44 fabricated
        {"limitations": []},
        {"corpus_patterns": ["both papers evaluate on narrow benchmarks"],
         "hypotheses": ["larger multilingual corpora may change conclusions"]},
    ])
    out = run_research_actions("GAP_EXPLORATION", "", ["a.pdf", "b.pdf"], _registry())
    md = out["markdown"]
    assert "### 1. Explicitly stated" in md and "small dataset" in md
    assert "p. 7" in md and "44" not in md.split("### 2")[0]   # fabricated page dropped
    assert "### 2. Patterns observed" in md and "narrow benchmarks" in md
    assert "### 3. Hypotheses (inferred" in md
    assert "not a claim about all published research" in md


# ---------------------------------------------------------- graph routing

def test_graph_routes_research_intent_through_actions(monkeypatch):
    from src.core.agent.graph import execute_agentic_pipeline

    monkeypatch.setattr(
        "src.core.generator.router.route_user_intent",
        lambda prompt, files: {"intent": "EVIDENCE_TABLE", "target_docs": ["a.pdf", "b.pdf"],
                               "focus": "methods and results"})
    monkeypatch.setattr("src.core.vector_store.get_registry", _registry)
    monkeypatch.setattr(
        "src.core.research_actions.run_research_actions",
        lambda intent, focus, targets, registry: {
            "actions": [{"tool": "evidence_table.extract", "args": {"doc": t},
                         "summary": "extraction"} for t in targets],
            "markdown": "## Research Evidence Table\n\n| Paper | Method |\n|---|---|",
            "data": []})

    def forbidden(*a, **kw):
        raise AssertionError("retrieval must not run for research intents")
    monkeypatch.setattr("src.core.vector_store.search_store", forbidden)
    monkeypatch.setattr("src.core.vector_store.search_image_store", forbidden)

    result = execute_agentic_pipeline("build an evidence table", ["a.pdf", "b.pdf"])

    assert result["intent"] == "EVIDENCE_TABLE"
    assert result["is_special_intent"] is True
    assert len(result["actions"]) == 2          # one logged action per paper
    assert "Research Evidence Table" in result["answer"]
    assert result["verification"]["status"] == "skipped"


def test_graph_research_intent_without_targets_degrades(monkeypatch):
    from src.core.agent.graph import execute_agentic_pipeline

    monkeypatch.setattr(
        "src.core.generator.router.route_user_intent",
        lambda prompt, files: {"intent": "GAP_EXPLORATION", "target_docs": [],
                               "focus": "gaps"})
    # Router upgrades untargeted research intents to STANDARD_CHAT in
    # understand_query, so this should flow through the standard path.
    monkeypatch.setattr("src.core.vector_store.search_store", lambda q, **kw: [])
    monkeypatch.setattr("src.core.vector_store.search_image_store", lambda q, **kw: [])
    monkeypatch.setattr("src.core.generator.answering.generate_answer",
                        lambda q, t, i, h: "best effort answer")

    result = execute_agentic_pipeline("what are the research gaps?", ["a.pdf"])
    # Untargeted research intent degrades to standard chat, not an error.
    assert result["intent"] == "STANDARD_CHAT"
    assert result["answer"]
