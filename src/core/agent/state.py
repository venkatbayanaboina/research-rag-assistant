"""Typed graph state for the agentic research pipeline.

Every node receives this state and returns a partial update. Only the fields
below are part of the contract between nodes; anything else must not be
keyed on.
"""

from typing import Annotated, Any, Dict, List
from typing_extensions import TypedDict


def _append_errors(left, right):
    """Reducer for the errors channel: node failures accumulate."""
    return (left or []) + (right or [])


class ResearchState(TypedDict, total=False):
    # Request identity
    run_id: str
    query: str
    indexed_docs: List[str]
    use_rerank: bool
    chat_history: List[Dict[str, str]]

    # Understanding & planning
    intent: str                    # STANDARD_CHAT | SUMMARY | SECTION_SUMMARY | COMPARISON
                                   # | EVIDENCE_TABLE | CONTRADICTION | GAP_EXPLORATION
    target_docs: List[str]
    focus: str                     # research focus for the Phase-5 research intents
    plan: Dict[str, Any]

    # Retrieval
    expanded_queries: List[str]    # refinement rewrites (original query first)
    text_results: List[Dict[str, Any]]
    image_results: List[Dict[str, Any]]
    # Speculative retrieval executed in parallel with the router LLM call
    # (standard intents consume it; special intents discard it)
    speculative: Dict[str, Any]

    # Control flow
    evidence_status: str           # sufficient | partial | insufficient
    refinement_iteration: int
    model_call_count: int
    deadline: float                # wall-clock time.time() deadline
    errors: Annotated[List[str], _append_errors]

    # Output
    answer: str
    is_special_intent: bool
    verification: Dict[str, Any]
    actions: List[Dict[str, Any]]  # observable agent actions (tool, args, seconds, summary)
    final: Dict[str, Any]          # the response dict returned to callers
