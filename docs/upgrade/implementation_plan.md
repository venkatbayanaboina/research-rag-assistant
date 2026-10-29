# Implementation Plan — Executed Version

**Note on process:** the original spec required stopping for approval after Phase 1. The owner explicitly waived the approval gate ("no permissions ask, take all decisions") on 2026-10-28, so this plan was executed immediately after being written. Each work item records its outcome status.

---

## W1 — Hybrid Retrieval: BM25 ⊕ Dense with RRF  ✅ DONE

- **Hypothesis:** exact-lexical evidence (dataset names, equation names, "Table 3", rare terms) is missed by dense-only retrieval; adding BM25 + RRF raises Recall@5/MRR on the 60-case gold set without hurting text-type questions.
- **Baseline:** dense-only Hit@1 53.33% / Recall@5 80.00% / MRR 0.6356 (measured, `baseline.md` §2.1).
- **Change (one variable):** add `mode="hybrid"` to `search_store` = RRF(k=60) of dense ranking and BM25Okapi ranking, before reranking. Config flag `HYBRID_RETRIEVAL_ENABLED` (default on; `dense` path unchanged). New `src/core/sparse_retrieval.py` + `src/core/fusion.py`.
- **Files:** `src/core/sparse_retrieval.py` (new), `src/core/fusion.py` (new), `src/core/vector_store.py`, `config.py`.
- **Acceptance:** unit tests for BM25 ranking & RRF fusion; benchmark re-run shows Recall@5 ≥ baseline (keep) — else revert flag default.
- **Rollback:** set `HYBRID_RETRIEVAL_ENABLED = False` (or `mode="dense"` per call).

## W2 — LangGraph Agentic Orchestration  ✅ DONE

- **Change:** `src/core/agent/` package implementing the graph in `architecture.md` §2: validation → understanding (existing router) → planning (deterministic) → retrieval (existing stores via hybrid mode) → sufficiency (deterministic; optional LLM) → bounded refinement (≤ `AGENT_MAX_REFINEMENTS`) → generation (existing) → verification → finalize. Typed `ResearchState`, per-node error capture, run IDs, model-call budget, deadline.
- **Files:** `src/core/agent/{__init__,state,nodes,graph}.py` (new), `src/core/generator/orchestrator.py` (delegation + fallback), `config.py`.
- **Acceptance:** deterministic transition tests with stubbed retrievers: sufficient→answer, insufficient→refine→budget-exhaust→qualified answer, router failure→deterministic fallback intent, retrieval exception→error state with usable response.
- **Rollback:** `AGENTIC_PIPELINE_ENABLED = False` restores the legacy path verbatim.

## W3 — Citation & Claim Verification  ✅ DONE

- **Change:** generation prompt requires inline `[Source N]` citations; `verification.py` Layer 1 checks marker resolution, page consistency, and citation coverage; Layer 2 optional LLM support verdicts (flag `VERIFY_CLAIMS_WITH_LLM`, default off). Verification report is returned in the result dict and rendered in the Streamlit expander.
- **Files:** `src/core/verification.py` (new), `src/core/generator/answering.py`, `app/components/chat.py` (render verification status).
- **Acceptance:** unit tests for valid markers, dangling marker, wrong-page marker, zero-marker answer (qualified), Layer-2 stub.

## W4 — Defect Fixes (from repository_audit.md)  ✅ DONE

- D1 `main.py compare` signature; D2 broken mock test updated to `docs_data` contract; D3 session-aware image registry in `get_key_images_for_doc`; D6 lock scope in `add_document_to_store` (hold `db_lock` across read-embed-write); D12 router output schema validation (enum + type checks).
- D4 (fabricated 96/98 fallback scores in `run_features_audit.py`) and D5 (README BM25 claim) are corrected in the evaluation/docs workstream (W6): the fallback now records `score: null` + error, and README §5 is rewritten to present only reproducible numbers.

## W5 — Tests & Suite Repair  ✅ DONE

- New: `tests/test_fusion.py`, `tests/test_sparse_retrieval.py`, `tests/test_verification.py`, `tests/test_graph.py` (stubbed-node transition tests), `tests/test_defect_fixes.py` (regression tests pinning D1/D3).
- Default run: `rag/bin/python3 -m pytest tests/test_features.py tests/test_fusion.py tests/test_sparse_retrieval.py tests/test_verification.py tests/test_graph.py tests/test_defect_fixes.py -q` — must pass 100%. `test_rag_eval.py` remains an opt-in integration test (needs live LLM).

## W6 — Benchmark, Evaluation Integrity, Docs  ✅ DONE

- `evaluation_suite/pipelines/run_retrieval_benchmark.py` — `--mode dense|hybrid --k 5`, per-type breakdown, JSON + Markdown output to `evaluation_suite/evaluations/retrieval_benchmark/`. Measured results in `docs/upgrade/evaluation_results.md`.
- `run_features_audit.py` fabricated-score fallback removed (D4).
- Docs: `repository_audit.md`, `baseline.md`, `architecture.md`, this file, `evaluation_results.md`, `security_review.md`, `decisions/ADR-001..004`; README rewritten where it claimed unreproducible metrics.

## W7 — Visual Retrieval: Evidence Rules (ADR-005)  ✅ DONE

- **Hypothesis:** the old visual search (fixed 0.28 CLIP cutoff, substring token matching, score-space mixing, hardcoded `[:2]` quota) returns irrelevant figures and pads answers to 2 images.
- **Baseline captured:** on a hi_res-ingested 21-page/21-figure PDF — "Show me Figure 1" (no such figure) → 2 wrong tables; CMRR query → right table + 1 noise table.
- **Change:** new `src/core/visual_ranking.py` (relative dense calibration, IDF-weighted token-SET caption matching, explicit-reference patterns, evidence qualification); `search_image_store` rewritten (also gains `session_paths`); agent/legacy pipelines consume the `explicit` flag and treat `K_IMAGE_RETRIEVAL` as a cap.
- **Measured (A/B, same store & queries):** "Show me Figure 1" 2→**0**; CMRR query 2 (1 wrong)→**1 (exact)**; genuinely visual query retained (2 relevant); text-only questions 0→0.
- **Acceptance:** 12 unit tests pin the rules (`tests/test_visual_ranking.py`), incl. the stray-digit and single-ubiquitous-token defects. **Rollback:** `git revert` the visual_ranking/vector_store changes; no config flag (the old behavior was a defect, not a feature).

## W8 — UI: Glassmorphism Redesign + UX Simplification (ADR-006)  ✅ DONE

- Creamy light glassmorphism design system centralized in `app/theme.py` (+ light theme in `.streamlit/config.toml`); hero, 3 metric cards, glass chat cards, verification chips, source cards, figure frames, suggestion-chip empty state.
- Text and decision load cut: strategy picker removed (hi_res only, `config.DEFAULT_PARSE_STRATEGY`), reranking always on (toggle removed), per-doc delete + Clear DB removed, replaced by a non-destructive **New Session** (store rotation; old data left for the 2-hour auto-purge). Destructive backend functions retained for CLI use.
- Sidebar header fix: Streamlit header kept as a transparent layer so the collapsed-sidebar reopen arrow (») stays reachable.
- Verified in-browser across empty / populated / live-cited-Q&A states.

## W9 — CLI Self-Healing Runtime  ✅ DONE

- `main.py` gains `ensure_runtime()`: when the current interpreter cannot import core deps (e.g. bare `python3 main.py ui` on system Python), the CLI re-executes itself through the bundled venv (guarded against re-exec loops; clear install guidance when no venv exists). Verified: `--help` untouched, `summarize` and `ui` auto-relaunch correctly.

## W10 — Agentic Research Actions (Phase 5 Features B/C/D, ADR-008)  ✅ DONE

- **Gap addressed:** the graph orchestrated but never acted — no tool selection, no Phase 5 research intelligence.
- **Change:** router extended to 7 intents (+ `EVIDENCE_TABLE`, `CONTRADICTION`, `GAP_EXPLORATION`, with `focus` extraction); new `execute_research_actions` graph node drives `src/core/research_actions.py` — evidence tables, contradiction classification (5 classes incl. "not a contradiction"), gap exploration with evidenced/pattern/hypothesis layers. Provenance guard: LLM page numbers validated against the registry; absent fields render "Not stated in retrieved evidence" (no guesses). Every action logged and surfaced in the UI "Agent actions" expander; bounded by the model-call budget, ≤5 papers.
- **Acceptance:** 8 new tests (provenance, no-guesses, classification schema, budget skip, gap labelling, graph routing with retrieval-forbidden assertion, untargeted degradation). **Live-verified:** real query → `intent=EVIDENCE_TABLE`, 3 LLM calls, action log populated, table rendered for both papers. Suite: **57 passed**.

## W11 — Evaluation Hardening (integrity-audit follow-up)  ✅ DONE

- **Bootstrap CIs** (`run_bootstrap_ci.py`, B=10k, seed 20260101): Hit@1 gain **within noise** (Δ +6.7pp, CI [0.0, 15.0]); Recall@5 and MRR gains **significant** (CIs exclude 0). Faithfulness 0.954 [0.906, 0.991], relevancy 0.893 [0.817, 0.955].
- **Failure analysis** (`docs/failure_analysis.md`): 5 retrieval misses, 19 wrong-ranks, 3 low-F, 7 low-R, 0 bad-gold, 0 generation errors, all examples listed.
- **Latency report** (`run_latency_report.py`): n=60 E2E median 14.24s / p95 57.92s / max 626.81s (provider stall kept), outliers named; retrieval-only median 94.5ms; hardware/provider stated.
- **Reproducibility**: `requirements.lock.txt` (210 pins), `Makefile` (`make eval` regenerates all report numbers from raw JSONs; `eval-live` re-runs live stages), `.env.example` refreshed, git-history secret scan clean (112M chars scanned), CI workflow (`.github/workflows/ci.yml`) running secret-scan + smoke + unit suite.
- **Hold-out scaffold** (`eval/`): JSONL schema, loader/validator, runner that REFUSES on empty set or missing frozen commit (both refusals verified by running it). Questions to be human-authored — none generated.
- **Review tooling**: `evaluation_suite/review/` CSVs (30 gold pairs, 20 answers) + `read_review_sheets.py` reporting gold error rate and judge-vs-human agreement once filled.
- **Independent judge config**: `JUDGE_MODEL` env / config option; full-60 run with Gemini 2.5 Flash written to `judge_results_independent.json`, reported separately.
- **Deployment**: Dockerfile + `docs/deployment.md` (local, Docker, Streamlit Cloud). Public link pending account setup.

## Risks & Limitations (carried forward)

- 60 QA pairs / 4 papers is a small benchmark; deltas < ~10pp are noise. Stated wherever results are quoted.
- Evidence-page matching is a proxy label; no human relevance judgments were collected.
- LLM-judge evaluations require API keys and are not part of the offline reproducible evidence.
- Visual retrieval qualification depends on caption/OCR quality; low-caption corpora attach fewer images (deliberately conservative — see ADR-005).
- Global model singletons and `db_lock` remain single-process assumptions (Streamlit server scope); multi-worker deployment is out of scope for this upgrade.
