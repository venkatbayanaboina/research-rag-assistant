# Baseline — Feasibility & Measured State (Phase 1)

**Date:** 2026-10-28 · Environment: `rag/` venv (Python 3.14.4), macOS arm64, device `mps`, repo at commit `38f3978` + uncommitted working-tree changes.

This file separates **verified facts**, **measured results**, **engineering assumptions**, and **unresolved questions**. Nothing here is estimated by feel; every number below was produced by running code against the repository in its pre-upgrade state.

---

## 1. Verified Facts

- The application, tests, and evaluation suite run under `rag/bin/python3` (3.14.4). Direct `rag/bin/<script>` invocation is broken (stale shebangs after a folder move); `rag/bin/python3 -m <module>` works.
- Model weights are fully cached in `~/.cache/huggingface/hub`: `BAAI/bge-large-en-v1.5`, `clip-ViT-B-32`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, `unstructuredio/yolo_x_layout`. **No network download is needed for local runs.**
- The global store holds **4,722 text chunks across 9 documents** and 6 image chunks. Chunk schema is not uniform: older chunks lack `section_title`.
- The gold QA dataset (`evaluation_suite/dataset/gold_qa_dataset.json`) covers 315 papers; **4 of them are indexed** (1407.7906, 1408.5882, 1409.3215, 1409.7770), yielding **60 usable QA pairs** (18 text, 18 equation, 18 figure, 6 table), each with an evidence page reference.
- `rank_bm25` and `langgraph` were absent; installed into `rag/` venv during Phase 1 (ADR-001). No existing package was upgraded.
- Importing `src.core.vector_store` outside Streamlit emits benign `streamlit` ScriptRunContext warnings (the module imports `streamlit` defensively).

## 2. Measured Results (pre-upgrade, unmodified code)

### 2.1 Retrieval quality baseline — dense-only, no reranker, k=5

Runner: inline script over the 60 gold QA pairs; relevance = retrieved chunk's `source_file` matches and `chunk.page == evidence.page`. FAISS `IndexFlatIP` over 4,722 normalized BGE vectors; BGE query prefix applied. Reranker intentionally **off** to isolate retrieval (not reranking) quality.

| Metric | Value |
|---|---|
| Hit@1 | **53.33%** (32/60) |
| Recall@5 | **80.00%** (48/60) |
| MRR | **0.6356** |
| Mean latency per query (embed + search, MPS, model warm) | **272.8 ms** |

Per question type:

| Type | n | Hit@1 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| text | 18 | 55.56% | 88.89% | 0.6722 |
| equation | 18 | 55.56% | 72.22% | 0.6083 |
| figure | 18 | 50.00% | 72.22% | 0.5972 |
| table | 6 | 50.00% | 100.00% | 0.7222 |

Raw JSON: `scratch/dense_baseline_results.json` (working notes) — canonical copies live in `evaluation_suite/evaluations/retrieval_benchmark/`.

### 2.2 Test suite status (pre-upgrade)

- `tests/test_features.py` — 3 tests, **2 pass / 1 fails**: `test_generate_comparison_mock` asserts the obsolete 4-argument `generate_comparison` signature (audit defect D2).
- `tests/test_rag_eval.py` — **not executable offline** (requires live LLM judge + API keys); skips only when no documents are indexed, which is not our case. Treated as an integration test, excluded from the default unit run.
- There are no tests for ingestion, chunking, vector store, fusion, reranking, routing, or orchestration.

### 2.3 Existing benchmark metrics (from repo docs) — reproducibility assessment

| Claimed (README §5) | Reproducible from this code? |
|---|---|
| 2×2 matrix "Dense + Sparse BM25" (31.06%→48.57% RAG score, Qwen2.5-3B judge) | **No** — no BM25 exists in the repo; no runner producing that table exists in `evaluation_suite/` |
| Gemini multimodal matrix 88.75% faithfulness / 83.75% relevancy (Llama-3.1-8B judge) | Not reproducible offline (needs DeepEval + live judge); scripts exist but datasets/results are not versioned |
| Features audit 98.33% overall (Gemini judge) | Reproducible only with the fabricated-fallback defect D4 (96–98% default on JSON parse failure), so the number is not trustworthy |

## 3. Engineering Assumptions (explicit, unverified)

1. Evidence-page match is a reasonable proxy for chunk relevance for this benchmark; section names in the gold dataset are not reliably reproduced by the parser, so only `page` is used.
2. LLM-as-judge scores are noisy; deterministic retrieval metrics are the primary regression signal for this upgrade. Judge-based scores will be reported separately and clearly labeled.
3. The `fast` parsing strategy's heuristic title detection (len < 150, no trailing period) is "good enough" for section tracking; improving it is out of scope unless the benchmark shows a chunking-attributable failure mode.
4. BM25 over 4,722 chunks rebuilds in well under a second, so query-time index construction with caching is acceptable (no persistence layer needed).
5. A max-2 refinement loop adds acceptable latency (~1 extra LLM round-trip only when evidence is judged insufficient).

## 4. Unresolved Questions

1. Whether the README §5.A BM25 numbers came from an unpushed experiment or another machine — no artifact in the repo can reproduce them. (Action taken: mark as non-reproducible; replace with measured numbers.)
2. Whether `unstructured`'s `hi_res` image-crop naming (`<file>_p<page>_img<n>.jpeg`) is stable across versions — relevant to the D7 ID-alignment defect.
3. Whether LangGraph's checkpointer (SQLite) is worth adding for a single-process Streamlit app — deferred; in-memory state suffices at current scale.
4. Token costs per query on Gemini 2.5 Flash — unmeasurable until usage metadata capture is added to the gates (planned in this upgrade).

## 5. Feasibility Verdict

- **Hybrid retrieval (BM25+dense+RRF): feasible offline**, measurable against the 60-case gold set, low risk (additive path, dense fallback preserved).
- **LangGraph orchestration: feasible** on Python 3.14 with `langgraph` installed; wrapped so that any graph failure falls back to the legacy pipeline.
- **Citation verification: feasible** deterministically (Layer 1) with an optional LLM Layer 2 behind a config flag.
- **Full LLM-judge evaluation: not feasible offline** — requires API keys; the deterministic retrieval benchmark + unit tests are the reproducible evidence base for this upgrade.
