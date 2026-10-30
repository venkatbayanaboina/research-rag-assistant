# ADR-008 — Agentic research actions (Phase 5 Features B, C, D)

**Date:** 2026-10-28 · **Status:** Accepted (live-verified end-to-end)

## Context

The LangGraph pipeline routed, retrieved, refined, and verified — but the model never *acted*: it chose no tools and executed no multi-step research work. Phase 5's research-intelligence features (B: evidence tables, C: contradiction analysis, D: gap exploration) were unimplemented, so calling the system "agentic" was only half-earned.

## Decision

The graph gains **tool-using research actions**, selected by the router and executed as bounded, observable pipelines:

1. **Router extended** to 7 intents: the original four plus `EVIDENCE_TABLE`, `CONTRADICTION`, `GAP_EXPLORATION`, each with a `focus` field (the topic/question to analyse). Schema-validated; unresolvable targets degrade to `STANDARD_CHAT` rather than erroring.
2. **New node `execute_research_actions`** (`src/core/research_actions.py`) runs the planned action sequence per intent:
   - **B — Evidence Table:** per-paper structured extraction (research question, method, dataset, metric, result, limitations) with evidence page references.
   - **C — Contradiction Analysis:** per-paper claims on the focus topic (with experimental conditions), then conflict classification into exactly `genuine_disagreement | different_datasets | different_evaluation_settings | different_definitions | not_a_contradiction`. Different numbers under different conditions are *reported as such*, never inflated into contradictions.
   - **D — Gap Exploration:** three clearly-separated layers — (1) limitations **explicitly stated** in the papers (with page citations), (2) patterns observed across the indexed corpus, (3) **hypotheses inferred by the system, labelled as such** — plus a scope disclaimer ("derived from N indexed papers only; not a claim about all published research").
3. **Provenance guard (no-guesses rule):** every page number an LLM produces is validated against the registry; fabricated pages are dropped, absent fields render as "Not stated in retrieved evidence" — never filled by guessing (spec Phase 5, Feature A/B requirement).
4. **Observable agency:** every tool invocation is logged (`{tool, args, seconds, summary|skipped|error}`), returned in the final result, and rendered in the UI's "Agent actions" expander. The `[AGENT]` run line and `plan.strategies` show the chosen action sequence.
5. **Bounded by the same budget:** research actions draw from `AGENT_MODEL_CALL_BUDGET` (routing spends 1; per-paper extractions spend the rest; skipped actions are logged, not silently dropped). Target scope capped at 5 papers.

## Live verification

`"Build an evidence table for the attention paper and the CNN sentence classification paper"` →
`intent=EVIDENCE_TABLE, llm_calls=3 (1 routing + 2 extractions), action log populated, evidence table rendered for both papers.` (289 s on saturated free OpenRouter models — extraction outputs are long; a fast provider collapses this, per ADR-007.)

## Tests

8 new tests (`tests/test_research_actions.py`): provenance page validation, no-guesses rendering, budget exhaustion logging, contradiction classification schema, "needs two claims" guard, gap-report labelling + scope disclaimer, graph routing for research intents (retrieval asserted NOT to run), and untargeted-intent degradation. Suite: **57 passed**.

## Consequences

- The system now earns "agentic": plan → act → observe → compose, with every action inspectable.
- Research actions add latency (per-paper LLM calls) — bounded by the budget and capped at 5 papers; acceptable for the deliberate multi-paper workflows they serve.
- Feature A (comparison) keeps its existing pipeline; Feature E (figure-aware research) is already served by retrieval + evidence rules.
