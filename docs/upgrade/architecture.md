# Target Architecture — Agentic Multimodal Research Intelligence

**Status:** Implemented per `implementation_plan.md` (approval gate waived by owner instruction, 2026-10-28).

## 1. Design Principles

1. **LangGraph is the control-flow layer only.** Retrieval, fusion, reranking, embedding, and generation remain the existing proven components. Nothing proven was replaced.
2. **Every new capability is flag-gated with a legacy fallback.** If LangGraph or a gate fails, the system degrades to the current pipeline, never to an error page.
3. **Provenance is never discarded.** All transformations preserve `{source_file, page, chunk_id/image_id, scores}`.
4. **Deterministic before LLM.** Sufficiency checks and citation verification are deterministic by default; LLM evaluators are opt-in flags.

## 2. Target Request Flow (as implemented)

```
User Query (Streamlit / CLI)
│
▼
Query Validation ──────────── deterministic: empty/length checks  [agent/nodes.py]
│
▼
Query Understanding ───────── existing LLM router (route_user_intent),
│                             hardened with schema validation + deterministic fallback
│
▼
Research Planning ─────────── deterministic bounded plan (no LLM call):
│                             objective, strategy set, budget, completion condition;
│                             research intents plan an explicit action sequence
│
▼
LangGraph Orchestrator (agent/graph.py) ─────────────────────────────────────
│
├─▶ Research Actions (EVIDENCE_TABLE / CONTRADICTION / GAP_EXPLORATION)
│     └─ execute_research_actions (research_actions.py):
│         per-paper structured extraction · conflict classification
│         (genuine / different conditions / not a contradiction) ·
│         evidenced limitations + patterns + labelled hypotheses.
│         Provenance guard: LLM page numbers validated vs registry,
│         absent fields render "Not stated" — never guessed (ADR-008).
│     → verify_response → finalize
│
├─▶ Retrieval Execution
│     ├─ Existing dense search  (vector_store.search_store, mode="dense")
│     ├─ NEW: BM25 sparse search (sparse_retrieval.search_store_sparse)
│     ├─ NEW: RRF fusion dense ⊕ sparse (k=60)  [config.HYBRID_RETRIEVAL_ENABLED]
│     ├─ Existing cross-encoder reranking (always on in the UI; config kill-switch)
│     └─ REBUILT: visual hybrid search (search_image_store → visual_ranking.py:
│           adaptive dense filter + IDF-weighted captions + evidence qualification;
│           returns 0..K_IMAGE_RETRIEVAL images — a cap, never a quota)
│
▼
Evidence Sufficiency Evaluation ── deterministic checks (chunk count, score floor,
│                                  target coverage); optional LLM judge (flag)
│
├─ Sufficient / Partial ──────▶ Answer Generation (existing generate_answer,
│                               upgraded citation-aware prompt)
│
├─ Insufficient ──────────────▶ Bounded Query Refinement (LLM rewrite, max
│                               AGENT_MAX_REFINEMENTS=2, budget-checked) ─▶ loop to Retrieval
│
└─ Budget exhausted ──────────▶ proceed with best evidence + uncertainty note
│
▼
Citation & Claim Verification (verification.py)
├─ Layer 1 deterministic: [Source N] markers resolve? pages match? coverage?
└─ Layer 2 LLM support check (optional, config.VERIFY_CLAIMS_WITH_LLM)
│
├─ Verified ─────────────▶ Final Research Result (status: verified)
├─ Unsupported claims ───▶ Final Research Result (status: qualified + issues listed)
└─ Verification failure ─▶ Final Research Result (status: failed → abstention notice)
│
▼
Final Research Report
{answer, intent, evidence, citations, verification, retrieval summary,
 evidence_status, refinement_iterations, run_id, model_call_count}
```

**Deviations from the spec diagram (justified):**
- *Research Planning is deterministic, not LLM-generated.* At this corpus scale a template plan per intent class yields the same routing decisions without an extra model call or a failure mode. The plan object is still produced and logged.
- *Refinement reuses the orchestrator's existing query-expansion LLM call* (bounded to 3 rewrites) instead of a separate rewrite prompt — one mechanism, one code path.
- *Checkpoints:* in-process state only (single-process Streamlit app; a persistent checkpointer adds a DB dependency the workload doesn't justify yet — see baseline.md §4.3).

## 3. Component Contracts

### 3.1 New modules

| Module | Contract |
|---|---|
| `src/core/sparse_retrieval.py` | `BM25Search` — tokenized BM25Okapi over the text registry, cached by `(chunks_path, mtime, count)`. `search_store_sparse(query, k, session_paths) → [{"score", "chunk"}]` |
| `src/core/fusion.py` | `rrf_fuse(rankings, k_constant=60) → fused [{"score", "chunk"}]` — pure function, unit-tested |
| `src/core/visual_ranking.py` | Pure visual-ranking logic: `build_caption_stats` (caption token DF), `sparse_visual_matches` (explicit "fig/table N" patterns + IDF-weighted caption evidence; token-SET matching, structural nouns & bare digits excluded), `qualify_visual` (evidence rules: explicit, dense∩sparse agreement, or strong-IDF caption evidence; dense-only rejected). Unit-tested against the A/B-observed failure modes |
| `src/core/verification.py` | `verify_citations(answer, evidence) → VerificationReport{status, issues[], citation_coverage, unsupported_claims[]}`; Layer 1 deterministic; Layer 2 optional LLM |
| `src/core/agent/state.py` | `ResearchState` TypedDict — the single typed graph state |
| `src/core/agent/nodes.py` | Node functions with explicit contracts: `validate_query`, `understand_query`, `plan_research`, `retrieve_evidence`, `evaluate_sufficiency`, `refine_query`, `generate_answer_node`, `verify_response`, `finalize` |
| `src/core/agent/graph.py` | `build_research_graph()` → compiled LangGraph; `execute_agentic_pipeline(prompt, indexed_docs, ...)` with legacy fallback |
| `src/core/research_actions.py` | Phase-5 agentic research tools (ADR-008): `run_research_actions(intent, focus, targets, registry)` executes evidence-table extraction, contradiction classification, and gap exploration — each an interleaved deterministic/LLM pipeline with provenance validation and an observable action log |
| `src/core/generator/router.py` | 7-intent router (4 chat + 3 research intents) with `focus` extraction, schema validation, deterministic fallback |
| `app/theme.py` | Central UI design system (creamy glassmorphism): stylesheet, hero/chip/source/figure HTML builders. Single place for visual retuning |
| `evaluation_suite/pipelines/run_retrieval_benchmark.py` | Reproducible dense/hybrid retrieval benchmark over the gold QA dataset |

### 3.2 Modified modules (additive where possible)

| Module | Change |
|---|---|
| `config.py` | New flags: `HYBRID_RETRIEVAL_ENABLED`, `AGENTIC_PIPELINE_ENABLED`, `AGENT_MAX_REFINEMENTS`, `AGENT_MODEL_CALL_BUDGET`, `AGENT_DEADLINE_SECONDS`, `EVIDENCE_MIN_CHUNKS`, `EVIDENCE_TARGET_COVERAGE`, `VERIFY_CLAIMS_WITH_LLM`, `DEFAULT_PARSE_STRATEGY="hi_res"`, and the visual-retrieval calibration set (`K_IMAGE_RETRIEVAL` as cap, `IMAGE_DENSE_ABS_FLOOR`, `IMAGE_DENSE_RELATIVE_MARGIN`, `IMAGE_SPARSE_MIN_IDF`, `IMAGE_SPARSE_STRONG_IDF`) |
| `src/core/vector_store.py` | `search_store(..., mode="dense"\|"hybrid")` — dense is byte-for-byte legacy behavior; hybrid = RRF(dense, BM25) before reranking. `search_image_store` rewritten around `visual_ranking.py` (evidence rules, 0..k results, `session_paths` support). Lock-scope fix: reentrant `db_lock` held across the whole read-embed-write ingestion sequence |
| `src/core/generator/orchestrator.py` | `execute_rag_pipeline` delegates to the agentic graph when `AGENTIC_PIPELINE_ENABLED`, falls back to legacy path on any graph error; visual boost keys on the `explicit` flag; image cap from config. Response dict gains optional fields, keeps all existing keys |
| `src/core/generator/answering.py` | Prompt requires inline `[Source N]` markers for factual claims (enables deterministic verification) |
| `src/core/generator/summarization.py` | `get_key_images_for_doc` uses session-aware paths (defect D3) |
| `src/core/generator/gates/openrouter.py` | Free-model rotation refreshed to live-verified slugs; overridable via `OPENROUTER_MODELS` env (defect D15) |
| `main.py` | `compare` subcommand fixed (D1); `ensure_runtime()` bootstrap re-executes the CLI through the bundled venv when the system interpreter lacks dependencies |
| `app/ui.py` | Glassmorphism hero + 3 live metric cards; theme injection |
| `app/components/sidebar.py` | Minimal: Upload → Library (glass doc cards) → New Session (non-destructive store rotation). Strategy picker, reranking toggle, per-doc delete and Clear DB removed (ADR-006) |
| `app/components/chat.py` | Empty state with suggestion chips, glass chat cards, verification chips, framed figures, source cards |

### 3.3 Error & budget model

- **Run ID** (uuid4 hex) assigned at graph entry; appears in every log line and in the final result.
- **Model-call budget:** routing (1) + expansion (1 per refinement, ≤3 calls total) + generation (1) + optional LLM verification (1). `model_call_count` is tracked in state; refinement is refused once `AGENT_MODEL_CALL_BUDGET` (default 12) is spent.
- **Deadline:** `AGENT_DEADLINE_SECONDS` (default 120) checked before each refinement iteration.
- **Error states:** each node catches its own exceptions into `state["errors"]`; the graph routes to `finalize` which always returns a usable response shape. A graph-level failure falls back to the legacy pipeline (logged as a fallback event).

## 4. What stays exactly as-is

PDF parsing (`ingestion.py`), chunking (`chunker.py`), BGE/CLIP embedding (`embedder.py`), the FAISS stores and dual-index model, cross-encoder reranker, gate failover chain (Gemini → OpenRouter → Ollama), DeepEval integration tests, and the evaluation-suite pipelines. The legacy `execute_rag_pipeline` code path remains callable and is the fallback. The Streamlit UI keeps its structure (ui.py + two components) but its presentation layer is the new glassmorphism design system.
