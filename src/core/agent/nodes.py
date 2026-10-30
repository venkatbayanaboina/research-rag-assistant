"""Node implementations for the research graph.

Contracts: every node takes the ResearchState and returns a dict of state
updates. Nodes catch their own expected failures into state["errors"] and
set routing fields; unexpected exceptions propagate to the graph executor,
which falls back to the legacy pipeline.
"""

import time
import uuid

import config
from src.core.generator.router import RESEARCH_INTENTS

SPECIAL_INTENTS = {"SUMMARY", "SECTION_SUMMARY", "COMPARISON"} | RESEARCH_INTENTS
VALID_INTENTS = {"STANDARD_CHAT"} | SPECIAL_INTENTS


def validate_query(state):
    """Deterministic input validation. Empty or oversized queries abort."""
    query = (state.get("query") or "").strip()
    updates = {"run_id": state.get("run_id") or uuid.uuid4().hex}
    if not query:
        updates["errors"] = ["Empty query rejected by validation."]
        updates["final"] = _abstain(
            updates["run_id"], "STANDARD_CHAT",
            "Please provide a non-empty research question."
        )
    elif len(query) > 4000:
        updates["query"] = query[:4000]
    return updates


def understand_query(state):
    """LLM intent routing with schema hardening and deterministic fallback.

    While the router LLM call is in flight, retrieval on the original query
    runs speculatively in a background thread: for standard intents its result
    replaces a serial retrieval pass, for special intents it is discarded.
    """
    import concurrent.futures

    from src.core.generator.router import route_user_intent

    use_rerank = state.get("use_rerank", config.RERANK_ENABLED)

    def _speculative():
        from src.core.vector_store import search_store, search_image_store
        try:
            return (search_store(state["query"], rerank=use_rerank),
                    search_image_store(state["query"]))
        except Exception:
            return [], []

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_speculative)

        updates = {}
        try:
            routing = route_user_intent(state["query"], state["indexed_docs"])
            intent = routing.get("intent", "STANDARD_CHAT")
            if intent not in VALID_INTENTS:
                intent = "STANDARD_CHAT"
            targets = [t for t in routing.get("target_docs", []) if t in state["indexed_docs"]]
            if intent in SPECIAL_INTENTS and not targets:
                intent = "STANDARD_CHAT"
            focus = routing.get("focus") or ""
            updates.update({"intent": intent, "target_docs": targets, "focus": focus})
        except Exception as e:
            updates.update({"intent": "STANDARD_CHAT", "target_docs": []})
            updates["errors"] = [f"Intent routing failed, defaulted to STANDARD_CHAT: {e}"]
        updates["model_call_count"] = state.get("model_call_count", 0) + 1

        if updates["intent"] == "STANDARD_CHAT":
            try:
                t_res, i_res = future.result(timeout=60)
                updates["speculative"] = {"text_results": t_res, "image_results": i_res}
            except Exception:
                updates["speculative"] = None
        else:
            updates["speculative"] = None
        return updates
    finally:
        executor.shutdown(wait=False)


def plan_research(state):
    """Deterministic bounded plan. No LLM call; produced for observability.
    Research intents (Phase 5) plan an explicit, observable action sequence."""
    intent = state.get("intent", "STANDARD_CHAT")
    special = intent in SPECIAL_INTENTS
    research_steps = {
        "EVIDENCE_TABLE": ["extract per-paper fields (method/dataset/metric/result/limitations)",
                           "validate evidence pages against the registry",
                           "render the evidence table"],
        "CONTRADICTION": ["extract each paper's claim on the focus topic",
                          "classify conflicts (genuine / different conditions / not a contradiction)"],
        "GAP_EXPLORATION": ["extract explicitly-stated limitations per paper",
                            "identify corpus patterns",
                            "formulate labelled hypotheses"],
    }
    plan = {
        "objective": {
            "SUMMARY": "Executive summary of the target documents",
            "SECTION_SUMMARY": "Section-by-section breakdown of the target document",
            "COMPARISON": "Side-by-side comparison matrix of the target documents",
            "EVIDENCE_TABLE": "Structured evidence table across the target papers",
            "CONTRADICTION": f"Conflict analysis on: {state.get('focus') or 'the papers’ findings'}",
            "GAP_EXPLORATION": "Research gaps across the indexed corpus",
        }.get(intent, "Answer the question strictly from retrieved evidence"),
        "requires_retrieval": not special,
        "strategies": (research_steps.get(intent, []) if intent in RESEARCH_INTENTS
                       else ([] if special else ["hybrid_text_dense_bm25_rrf", "visual_clip_hybrid"])),
        "expected_output": ("evidence_table" if intent == "EVIDENCE_TABLE" else
                            "conflict_classification" if intent == "CONTRADICTION" else
                            "gap_report" if intent == "GAP_EXPLORATION" else
                            "comparison_table" if intent == "COMPARISON" else "grounded_answer"),
        "verification": ("provenance pages validated in-action" if intent in RESEARCH_INTENTS
                         else "citation_integrity_layer1" if not special else "skipped_special_intent"),
        "completion_condition": (
            "planned research actions executed" if intent in RESEARCH_INTENTS else
            "special intent generated" if special else
            "evidence sufficient/partial, refinement budget exhausted, or deadline reached"
        ),
        "max_refinements": config.AGENT_MAX_REFINEMENTS,
    }
    return {"plan": plan}


def execute_research_actions(state):
    """Runs the planned research actions (Phase 5 Features B/C/D) via
    src/core/research_actions.py: per-paper structured extraction, provenance
    validation, and an observable action log. The composed report becomes the
    answer; verification is performed in-action (page validation)."""
    from src.core.vector_store import get_registry
    from src.core.research_actions import run_research_actions

    intent = state["intent"]
    targets = state.get("target_docs") or []
    try:
        registry = get_registry()
        if not targets:
            return {"answer": "I could not identify which indexed papers to analyse. "
                              "Name the papers in your request.",
                    "is_special_intent": True, "actions": [],
                    "errors": [f"{intent} ran without resolvable target documents."]}
        result = run_research_actions(intent, state.get("focus", ""), targets, registry)
        return {"answer": result["markdown"],
                "is_special_intent": True,
                "actions": result["actions"],
                "model_call_count": state.get("model_call_count", 0)
                                    + sum(1 for a in result["actions"] if not a.get("skipped")),
                }
    except Exception as e:
        return {"answer": f"Error running the {intent} research actions: {e}",
                "is_special_intent": True, "actions": state.get("actions") or [],
                "errors": [f"Research actions failed: {e}"]}


def retrieve_evidence(state):
    """Multi-query retrieval over the existing stores, fused with RRF.

    On the first pass, reuses the speculative results that ran in parallel
    with the router call (they are already filtered for the resolved target
    docs here). Refinement passes re-retrieve with batch-embedded queries.
    Retrieval failures degrade to empty evidence so the graph can route to
    refinement/abstention instead of crashing the run.
    """
    from src.core.vector_store import search_store, search_image_store
    from src.core.fusion import rrf_fuse

    queries = [state["query"]] + [q for q in state.get("expanded_queries", []) if q]
    targets = set(state.get("target_docs") or [])
    use_rerank = state.get("use_rerank", config.RERANK_ENABLED)

    def _filter(res):
        if targets:
            return [r for r in res if r["chunk"]["source_file"] in targets]
        return res

    speculative = state.get("speculative")
    if speculative and not state.get("expanded_queries"):
        # First pass: consume the parallel speculative retrieval.
        try:
            text_results = rrf_fuse([_filter(speculative.get("text_results") or [])],
                                    top_k=config.K_FINAL_CONTEXT)
            image_results = _fuse_image_results([_filter(speculative.get("image_results") or [])])
            return {"text_results": text_results, "image_results": image_results,
                    "speculative": None}
        except Exception as e:
            return {"text_results": [], "image_results": [],
                    "errors": [f"Speculative retrieval consumption failed: {e}"],
                    "speculative": None}

    text_rankings, image_rankings = [], []
    try:
        if len(queries) > 1:
            # Batch-embed all queries in one forward pass.
            from src.core.embedder import embed_queries
            vectors = embed_queries(queries)
            for q, vec in zip(queries, vectors):
                t_res = search_store(q, rerank=use_rerank, query_vector=vec)
                text_rankings.append(_filter(t_res))
        else:
            t_res = search_store(queries[0], rerank=use_rerank)
            text_rankings.append(_filter(t_res))

        for q in queries:
            i_res = search_image_store(q)
            image_rankings.append(_filter(i_res))
    except Exception as e:
        return {"text_results": [], "image_results": [],
                "errors": [f"Retrieval execution failed: {e}"], "speculative": None}

    text_results = rrf_fuse(text_rankings, top_k=config.K_FINAL_CONTEXT)
    image_results = _fuse_image_results(image_rankings)

    return {"text_results": text_results, "image_results": image_results,
            "speculative": None}


def _fuse_image_results(query_rankings):
    """RRF over per-query visual rankings, boosting explicit figure/table
    references from the original query. The final count is capped by
    config.K_IMAGE_RETRIEVAL — search_image_store may already return fewer
    (or zero) qualified images."""
    RRF_K = config.RRF_K_CONSTANT
    scores, objects, order = {}, {}, []
    for query_idx, ranking in enumerate(query_rankings):
        for rank_0, res in enumerate(ranking):
            chunk = res["chunk"]
            key = (chunk.get("source_file", ""), chunk.get("page"), chunk.get("image_path", ""))
            if key not in scores:
                scores[key] = 0.0
                objects[key] = chunk
                order.append(key)
            contrib = 1.0 / (RRF_K + rank_0 + 1)
            if query_idx == 0 and res.get("explicit"):
                contrib += 2.0
            scores[key] += contrib
    fused = [{"score": scores[k], "chunk": objects[k]} for k in order]
    fused.sort(key=lambda item: item["score"], reverse=True)
    return fused[:config.K_IMAGE_RETRIEVAL]


def evaluate_sufficiency(state):
    """Deterministic evidence-sufficiency gate. LLM judges are deliberately
    not used here; the checks below are the practical deterministic core."""
    text_results = state.get("text_results") or []
    targets = state.get("target_docs") or []

    if not text_results:
        return {"evidence_status": "insufficient"}

    # With the cross-encoder active, scores are relevance logits: <= 0 means
    # the reranker judged even the top candidate irrelevant to the query.
    if state.get("use_rerank", config.RERANK_ENABLED):
        top_score = text_results[0].get("score", 0.0)
        if top_score <= 0:
            return {"evidence_status": "insufficient"}

    if len(text_results) < max(1, config.EVIDENCE_MIN_CHUNKS):
        return {"evidence_status": "partial"}

    if targets:
        present = {r["chunk"]["source_file"] for r in text_results}
        coverage = len(present & set(targets)) / len(targets)
        if coverage + 1e-9 < config.EVIDENCE_TARGET_COVERAGE:
            return {"evidence_status": "partial"}

    return {"evidence_status": "sufficient"}


def refine_query(state):
    """Bounded refinement: LLM rewrites the query, then retrieval re-runs."""
    from src.core.generator.orchestrator import generate_similar_queries

    iteration = state.get("refinement_iteration", 0) + 1
    updates = {"refinement_iteration": iteration}

    out_of_budget = state.get("model_call_count", 0) >= config.AGENT_MODEL_CALL_BUDGET
    past_deadline = time.time() > state.get("deadline", float("inf"))
    if out_of_budget or past_deadline:
        reason = "model-call budget exhausted" if out_of_budget else "deadline reached"
        updates["errors"] = [f"Refinement skipped: {reason}."]
        updates["evidence_status"] = "partial"
        return updates

    try:
        rewrites = generate_similar_queries(state["query"])[:3]
        updates["expanded_queries"] = rewrites
        updates["model_call_count"] = state.get("model_call_count", 0) + (1 if rewrites else 0)
        if not rewrites:
            updates["errors"] = ["Refinement produced no rewrites; proceeding with current evidence."]
            updates["evidence_status"] = "partial"
    except Exception as e:
        updates["errors"] = [f"Refinement failed: {e}"]
        updates["evidence_status"] = "partial"
    return updates


def generate_answer_node(state):
    """Grounded answer generation from the fused evidence."""
    from src.core.generator.answering import generate_answer

    try:
        answer = generate_answer(
            state["query"],
            state.get("text_results") or [],
            state.get("image_results") or [],
            state.get("chat_history") or [],
        )
        return {
            "answer": answer,
            "is_special_intent": False,
            "model_call_count": state.get("model_call_count", 0) + 1,
        }
    except Exception as e:
        return {
            "answer": f"Error generating answer: {e}",
            "is_special_intent": False,
            "errors": [f"Answer generation failed: {e}"],
        }


def generate_special_node(state):
    """Direct-generation path for summary/section-summary/comparison intents.
    Mirrors the legacy dispatch in orchestrator.execute_rag_pipeline."""
    from src.core.vector_store import get_registry
    from src.core.generator import generate_summary, generate_section_summaries
    from src.core.generator.comparison import generate_comparison

    intent = state["intent"]
    targets = state.get("target_docs") or []
    try:
        registry = get_registry()
        if intent == "COMPARISON":
            docs_data = [
                {"name": doc, "chunks": [c for c in registry if c["source_file"] == doc]}
                for doc in targets
            ]
            docs_data = [d for d in docs_data if d["chunks"]]
            if len(docs_data) >= 2:
                answer = generate_comparison(docs_data)
            else:
                answer = (f"Error: Could only retrieve chunks for {len(docs_data)} of the "
                          f"requested documents. (Need at least 2).")
        elif intent == "SECTION_SUMMARY":
            doc_chunks = [c for c in registry if c["source_file"] == targets[0]]
            answer = generate_section_summaries(doc_chunks) if doc_chunks else \
                f"Error: No text chunks found in registry for document '{targets[0]}'."
        else:  # SUMMARY
            summaries = []
            for doc in targets:
                doc_chunks = [c for c in registry if c["source_file"] == doc]
                if doc_chunks:
                    summaries.append(f"### Summary of {doc}\n{generate_summary(doc_chunks)}")
                else:
                    summaries.append(f"Error: No text chunks found in registry for document '{doc}'.")
            answer = "\n\n---\n\n".join(summaries)
        return {"answer": answer, "is_special_intent": True,
                "model_call_count": state.get("model_call_count", 0) + 1}
    except Exception as e:
        return {"answer": f"Error generating {intent.lower()}: {e}", "is_special_intent": True,
                "errors": [f"Special-intent generation failed: {e}"]}


def verify_response(state):
    """Citation integrity gate. Special intents skip verification (their
    outputs are structural tables/summaries, not cited claims)."""
    from src.core.verification import verify_citations

    if state.get("is_special_intent"):
        return {"verification": {"status": "skipped", "issues": [], "citation_coverage": None,
                                 "unsupported_claims": []}}

    report = verify_citations(state.get("answer", ""), state.get("text_results") or [])
    return {"verification": report}


def finalize(state):
    """Builds the final response dict. Shape is a superset of the legacy
    pipeline result so all existing callers keep working."""
    from src.core.verification import format_verification_note

    if state.get("final"):
        return {}

    answer = state.get("answer", "")
    verification = state.get("verification") or {"status": "skipped"}
    if verification.get("status") in ("qualified", "failed"):
        answer = answer + format_verification_note(verification)

    return {"final": {
        "intent": state.get("intent", "STANDARD_CHAT"),
        "answer": answer,
        "image_results": state.get("image_results") or [],
        "search_results": state.get("text_results") or [],
        "is_special_intent": state.get("is_special_intent", False),
        "run_id": state.get("run_id"),
        "evidence_status": state.get("evidence_status", "unknown"),
        "verification": verification,
        "refinement_iterations": state.get("refinement_iteration", 0),
        "model_call_count": state.get("model_call_count", 0),
        "plan": state.get("plan"),
        "actions": state.get("actions") or [],
        "errors": state.get("errors") or [],
    }}


def _abstain(run_id, intent, message):
    """Terminal response for rejected queries."""
    return {
        "intent": intent,
        "answer": message,
        "image_results": [],
        "search_results": [],
        "is_special_intent": True,
        "run_id": run_id,
        "evidence_status": "abstained",
        "verification": {"status": "skipped"},
        "refinement_iterations": 0,
        "model_call_count": 0,
        "plan": None,
        "errors": [],
    }
