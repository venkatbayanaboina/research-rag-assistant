# ADR-003 — LangGraph orchestration with deterministic-first gates

**Date:** 2026-10-28 · **Status:** Accepted

## Context

Control flow lived inside `orchestrator.execute_rag_pipeline` as imperative code: no evidence-sufficiency gate, no bounded refinement, no citation checking, no run identity, no budget. The spec requires LangGraph orchestration with those capabilities while preserving existing components.

## Decision

1. **Graph topology** per `architecture.md` §2: validate → understand (existing LLM router) → plan (deterministic) → retrieve (existing stores) → sufficiency (deterministic) → refine (bounded ≤2) → generate (existing) → verify → finalize. Special intents (SUMMARY/SECTION_SUMMARY/COMPARISON) branch straight to generation, matching legacy behavior.
2. **Deterministic before LLM.** Sufficiency uses: non-empty evidence, top cross-encoder logit > 0 (when reranking is on), minimum chunk count, and target-document coverage. No LLM judge in the loop — it adds a failure mode and cost without a measured need at this corpus scale.
3. **Verification** is deterministic Layer 1 by default; LLM Layer 2 (`VERIFY_CLAIMS_WITH_LLM`) is opt-in and advisory-only.
4. **The legacy pipeline is retained verbatim** as `_execute_legacy_pipeline` and is the fallback for any graph-level failure. `AGENTIC_PIPELINE_ENABLED=False` bypasses the graph entirely.
5. **Generation prompt now requires inline `[Source N]` markers**, making Layer-1 verification meaningful. This is the only behavioral change to existing prompts.

## Consequences

- Response dict is a superset of the legacy shape (all existing keys preserved; new keys: `run_id`, `evidence_status`, `verification`, `refinement_iterations`, `model_call_count`, `plan`, `errors`). Streamlit and CLI callers work unchanged.
- Transition tests (`tests/test_graph.py`) stub all LLM/retrieval dependencies and run offline.
- Known trade-off: when all LLM gates fail (e.g., free-tier quota exhausted), the graph still returns a usable response whose text is the generation error message, flagged `qualified` by verification. Abstention wording for this case could be friendlier; deferred until measured user impact.
