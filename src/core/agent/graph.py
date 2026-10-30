"""LangGraph research graph and its public entry point.

Graph topology:

    START -> validate_query
    validate_query -> understand_query            | finalize (rejected input)
    understand_query -> plan_research
    plan_research -> retrieve_evidence            | generate_special (special intents)
    retrieve_evidence -> evaluate_sufficiency
    evaluate_sufficiency -> generate_answer_node  | refine_query | generate_answer_node (budget out)
    refine_query -> retrieve_evidence
    generate_answer_node -> verify_response
    generate_special -> finalize
    verify_response -> finalize

`execute_agentic_pipeline` never raises: any graph-level failure is logged and
the caller receives `None`, which tells the orchestrator to use the legacy
pipeline instead.
"""

import time

import config


def build_research_graph():
    from langgraph.graph import StateGraph, START, END
    from src.core.agent.state import ResearchState
    from src.core.agent import nodes

    graph = StateGraph(ResearchState)

    graph.add_node("validate_query", nodes.validate_query)
    graph.add_node("understand_query", nodes.understand_query)
    graph.add_node("plan_research", nodes.plan_research)
    graph.add_node("retrieve_evidence", nodes.retrieve_evidence)
    graph.add_node("evaluate_sufficiency", nodes.evaluate_sufficiency)
    graph.add_node("refine_query", nodes.refine_query)
    graph.add_node("generate_answer_node", nodes.generate_answer_node)
    graph.add_node("generate_special", nodes.generate_special_node)
    graph.add_node("execute_research_actions", nodes.execute_research_actions)
    graph.add_node("verify_response", nodes.verify_response)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "validate_query")
    graph.add_conditional_edges(
        "validate_query",
        lambda s: "finalize" if s.get("final") else "understand_query",
        {"finalize": "finalize", "understand_query": "understand_query"},
    )
    graph.add_edge("understand_query", "plan_research")
    graph.add_conditional_edges(
        "plan_research",
        lambda s: ("execute_research_actions" if s.get("intent") in nodes.RESEARCH_INTENTS
                   else "generate_special" if s.get("intent") in nodes.SPECIAL_INTENTS
                   else "retrieve_evidence"),
        {"execute_research_actions": "execute_research_actions",
         "generate_special": "generate_special",
         "retrieve_evidence": "retrieve_evidence"},
    )
    graph.add_edge("retrieve_evidence", "evaluate_sufficiency")
    graph.add_conditional_edges(
        "evaluate_sufficiency",
        _route_after_sufficiency,
        {"refine_query": "refine_query", "generate_answer_node": "generate_answer_node"},
    )
    graph.add_edge("refine_query", "retrieve_evidence")
    graph.add_edge("generate_answer_node", "verify_response")
    graph.add_edge("generate_special", "finalize")
    graph.add_edge("execute_research_actions", "verify_response")
    graph.add_edge("verify_response", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


def _route_after_sufficiency(state):
    """Sufficient/partial evidence generates now; insufficient evidence gets a
    bounded refinement pass unless the budget or deadline is already gone."""
    if state.get("evidence_status") in ("sufficient", "partial"):
        return "generate_answer_node"
    if state.get("refinement_iteration", 0) >= config.AGENT_MAX_REFINEMENTS:
        return "generate_answer_node"
    if state.get("model_call_count", 0) >= config.AGENT_MODEL_CALL_BUDGET:
        return "generate_answer_node"
    if time.time() > state.get("deadline", float("inf")):
        return "generate_answer_node"
    return "refine_query"


_compiled = None


def get_compiled_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_research_graph()
    return _compiled


def execute_agentic_pipeline(prompt, indexed_docs, use_rerank=None, chat_history=None):
    """Runs the research graph. Returns the final result dict, or None if the
    graph itself failed (caller falls back to the legacy pipeline)."""
    if use_rerank is None:
        use_rerank = config.RERANK_ENABLED

    initial_state = {
        "run_id": time.strftime("%H%M%S") + "-" + str(int(time.time() * 1000) % 100000),
        "query": prompt,
        "indexed_docs": list(indexed_docs or []),
        "use_rerank": use_rerank,
        "chat_history": list(chat_history or []),
        "intent": "STANDARD_CHAT",
        "target_docs": [],
        "refinement_iteration": 0,
        "model_call_count": 0,
        "expanded_queries": [],
        "errors": [],
        "deadline": time.time() + config.AGENT_DEADLINE_SECONDS,
    }

    try:
        result = get_compiled_graph().invoke(initial_state)
        final = result.get("final")
        if not final:
            raise RuntimeError("graph finished without a final response")
        print(f"[AGENT] run={final.get('run_id')} intent={final.get('intent')} "
              f"evidence={final.get('evidence_status')} refinement="
              f"{final.get('refinement_iterations')} llm_calls={final.get('model_call_count')} "
              f"verification={final.get('verification', {}).get('status')}")
        return final
    except Exception as e:
        print(f"[AGENT] Graph execution failed: {e}")
        return None
