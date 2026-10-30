# ADR-002 — Hybrid text retrieval: BM25 ⊕ dense fused by RRF

**Date:** 2026-10-28 · **Status:** Accepted (kept; benchmark evidence below)

## Context

Text retrieval was dense-only (BGE + FAISS FlatIP). Measured baseline on the 60-case gold benchmark: Hit@1 53.33%, Recall@5 80.00%, MRR 0.6356. The README already claimed a "Dense + Sparse BM25" hybrid that did not exist (audit defect D5).

## Hypothesis

Exact-lexical evidence (dataset names, metric names, "Table 3", rare tokens) is missed by dense embeddings; adding a BM25 ranking fused with the dense ranking via Reciprocal Rank Fusion improves recall and rank quality without a latency penalty that matters.

## Change (one variable)

`search_store(query, ..., mode="hybrid")`: dense top-K ranking (unchanged) fused with a BM25Okapi ranking over the same registry via `rrf_fuse` (k=60), then the existing cross-encoder reranking. `mode="dense"` preserves the pre-upgrade path byte-for-byte. Default controlled by `config.HYBRID_RETRIEVAL_ENABLED` (on).

## Measured result (same 60 cases, rerank off, k=5)

| Metric | Dense | Hybrid | Δ |
|---|---:|---:|---:|
| Hit@1 | 53.33% | **60.00%** | +6.67 pp |
| Recall@5 | 80.00% | **91.67%** | +11.67 pp |
| MRR | 0.6356 | **0.7367** | +0.1011 |
| Mean latency/query | 230.0 ms | 258.6 ms | +28.6 ms |

Per-type Recall@5: equation 72.2→83.3%, figure 72.2→94.4%, text 88.9→94.4%, table 100→100%.

## Decision

**Keep** hybrid as the default. All question types improved or held; the largest gains are exactly where dense retrieval was weakest (figures/equations with precise lexical anchors). Latency increase (~29 ms) is negligible next to embedding + LLM costs.

## Caveats

- 60 cases / 4 papers is small; deltas ≳10 pp are the meaningful threshold. The improvement is uniform across types, which supports the hypothesis.
- BM25 index is rebuilt in-process when the registry file changes (mtime+size key); no persistence layer was added.

**Rollback:** `HYBRID_RETRIEVAL_ENABLED = False` (or `mode="dense"` per call).
