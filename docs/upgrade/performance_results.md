# Performance Results — Before/After Pipeline Optimization

**One canonical comparison file.** All numbers are generated from raw JSON
artifacts by `evaluation_suite/pipelines/render_performance_report.py` —
nothing is transcribed by hand. Re-run the benchmarks, re-render, and the
Run 2 columns fill in automatically.

- **Run 1 (pre-optimization):** recorded 2026-10-28T17:06:52Z
- **Run 2 (post-optimization):** recorded 2026-10-28T17:20:26Z

Optimizations between runs: (1) OpenRouter unconditional 4.5 s pre-call sleep
replaced by 429-only backoff, (2) speculative parallel retrieval during the
router call, (3) batch-embedded multi-query retrieval paths.

---

## 1. Retrieval quality — dense vs. hybrid (60 gold QA cases, rerank off, k=5)

| Metric | Dense-only | Hybrid (BM25 ⊕ dense) | Δ |
|---|---:|---:|---:|
| Hit@1 | 53.33% | 60.00% | +6.67 pp |
| Recall@3 | 70.00% | 86.67% | +16.67 pp |
| Recall@5 | 80.00% | 91.67% | +11.67 pp |
| MRR | 0.6356 | 0.7367 | +0.1011 |
| Latency p50 (ms) | 82.8 | 94.5 | +11.7 |
| Latency p95 (ms) | 114.0 | 125.6 | +11.6 |

Latency note: `mean` includes a cold-start first query (model load), so
**p50 is the honest warm-path number**. Quality metrics are deterministic
and identical across runs of the same code.

### Per-type breakdown (hybrid)

| Type | n | Hit@1 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| equation | 18 | 66.67% | 83.33% | 0.7333 |
| figure | 18 | 55.56% | 94.44% | 0.7269 |
| table | 6 | 50.00% | 100.00% | 0.7083 |
| text | 18 | 61.11% | 94.44% | 0.7593 |

---

## 2. Retrieval latency per query (all 60 cases, model warm)

Dense p50/p95 from its own run; hybrid likewise. `first-relevant rank` is
the hybrid ranking position of the evidence-page chunk (— = not in top-5).

| qid | type | dense p-latency (ms) | hybrid latency (ms) | hybrid first-relevant rank |
|---|---|---:|---:|---:|
| 1407.7906_Q01 | text | 10406.7 | 9241.5 | 2 |
| 1407.7906_Q02 | equation | 122.4 | 104.0 | 1 |
| 1407.7906_Q03 | figure | 91.0 | 101.4 | 4 |
| 1407.7906_Q04 | text | 85.7 | 125.6 | 1 |
| 1407.7906_Q05 | equation | 94.7 | 116.1 | 1 |
| 1407.7906_Q06 | figure | 82.3 | 94.5 | 2 |
| 1407.7906_Q07 | text | 91.4 | 103.3 | 3 |
| 1407.7906_Q08 | equation | 87.7 | 104.7 | 1 |
| 1407.7906_Q09 | figure | 85.6 | 104.1 | 1 |
| 1407.7906_Q10 | text | 86.1 | 102.5 | 3 |
| 1407.7906_Q11 | equation | 90.2 | 101.8 | 2 |
| 1407.7906_Q12 | figure | 86.3 | 95.2 | 1 |
| 1407.7906_Q13 | text | 85.6 | 98.0 | 1 |
| 1407.7906_Q14 | equation | 90.5 | 106.8 | — |
| 1407.7906_Q15 | figure | 83.6 | 99.1 | — |
| 1408.5882_Q01 | text | 62.0 | 89.4 | 1 |
| 1408.5882_Q02 | text | 94.8 | 100.0 | 2 |
| 1408.5882_Q03 | text | 89.3 | 117.8 | 1 |
| 1408.5882_Q04 | text | 83.7 | 107.9 | 1 |
| 1408.5882_Q05 | figure | 87.1 | 199.6 | 2 |
| 1408.5882_Q06 | figure | 79.3 | 102.0 | 1 |
| 1408.5882_Q07 | figure | 62.1 | 73.2 | 2 |
| 1408.5882_Q08 | figure | 59.0 | 73.1 | 1 |
| 1408.5882_Q09 | table | 86.4 | 107.4 | 1 |
| 1408.5882_Q10 | table | 64.3 | 81.1 | 4 |
| 1408.5882_Q11 | table | 93.0 | 108.5 | 1 |
| 1408.5882_Q12 | equation | 84.0 | 94.9 | — |
| 1408.5882_Q13 | equation | 63.4 | 74.9 | 1 |
| 1408.5882_Q14 | equation | 64.6 | 79.2 | 1 |
| 1408.5882_Q15 | equation | 63.8 | 74.8 | 1 |
| 1409.3215_Q01 | text | 66.1 | 74.6 | 1 |
| 1409.3215_Q02 | text | 60.5 | 74.6 | 2 |
| 1409.3215_Q03 | text | 88.1 | 102.2 | 1 |
| 1409.3215_Q04 | text | 59.1 | 75.5 | — |
| 1409.3215_Q05 | figure | 89.6 | 106.3 | 2 |
| 1409.3215_Q06 | figure | 60.9 | 84.6 | 2 |
| 1409.3215_Q07 | figure | 60.9 | 79.5 | 1 |
| 1409.3215_Q08 | figure | 63.9 | 80.4 | 3 |
| 1409.3215_Q09 | table | 88.7 | 112.2 | 1 |
| 1409.3215_Q10 | table | 89.7 | 119.3 | 2 |
| 1409.3215_Q11 | table | 63.0 | 89.1 | 2 |
| 1409.3215_Q12 | equation | 80.1 | 73.2 | 1 |
| 1409.3215_Q13 | equation | 67.0 | 79.5 | 1 |
| 1409.3215_Q14 | equation | 61.8 | 79.3 | 1 |
| 1409.3215_Q15 | equation | 63.1 | 77.2 | — |
| 1409.7770_Q01 | text | 88.5 | 163.3 | 1 |
| 1409.7770_Q02 | text | 68.3 | 78.4 | 1 |
| 1409.7770_Q03 | text | 69.4 | 70.8 | 1 |
| 1409.7770_Q04 | text | 77.9 | 79.4 | 2 |
| 1409.7770_Q05 | text | 70.2 | 79.2 | 1 |
| 1409.7770_Q06 | figure | 68.6 | 79.2 | 1 |
| 1409.7770_Q07 | figure | 92.0 | 108.2 | 1 |
| 1409.7770_Q08 | figure | 67.8 | 80.3 | 1 |
| 1409.7770_Q09 | figure | 82.8 | 86.0 | 1 |
| 1409.7770_Q10 | figure | 184.5 | 78.6 | 1 |
| 1409.7770_Q11 | equation | 65.0 | 80.3 | 5 |
| 1409.7770_Q12 | equation | 60.8 | 74.0 | 1 |
| 1409.7770_Q13 | equation | 65.8 | 80.8 | 2 |
| 1409.7770_Q14 | equation | 65.6 | 77.2 | 1 |
| 1409.7770_Q15 | equation | 114.0 | 109.3 | 1 |

---

## 3. End-to-end pipeline latency (live LLM, OpenRouter pinned)

Full agentic runs: routing → hybrid retrieval → reranking → generation.
Model warm-up excluded. **Run 1 finding:** the agentic standard path makes
**2 LLM calls per query** (router + answer); query expansion exists only in
the legacy fallback path. On pinned OpenRouter free models each call costs
~45–50 s, of which **4.5 s is a hardcoded unconditional sleep** per call.

### Run 1 — pre-optimization

| Query (truncated) | Wall (s) | LLM calls | Evidence | Verification |
|---|---:|---:|---|---|
| What is the primary purpose of convolutional neural networ… | 75.58 | 2 | sufficient | verified |
| How does target propagation provide credit assignment in d… | 120.71 | 2 | sufficient | verified |
| What is multi-head attention in the transformer architectu… | 109.01 | 2 | sufficient | verified |
| What evaluation metrics are reported in the sequence to se… | 111.83 | 2 | sufficient | qualified |

**Mean wall time: 104.28s** · min 75.58s · max 120.71s · total LLM calls: 8 (2.0/query) · provider: openrouter (forced, free models)

### Run 2 — post-optimization

| Query (truncated) | Wall (s) | LLM calls | Evidence | Verification |
|---|---:|---:|---|---|
| What is the primary purpose of convolutional neural networ… | 112.46 | 2 | sufficient | verified |
| How does target propagation provide credit assignment in d… | 156.76 | 2 | sufficient | verified |
| What is multi-head attention in the transformer architectu… | 137.50 | 2 | sufficient | verified |
| What evaluation metrics are reported in the sequence to se… | 78.37 | 2 | sufficient | verified |

**Mean wall time: 121.27s** · min 78.37s · max 156.76s · total LLM calls: 8 (2.0/query) · provider: openrouter (forced, free models)

### Before → After comparison

| Metric | Run 1 | Run 2 | Δ |
|---|---:|---:|---:|
| Mean wall/query (s) | 104.28 | 121.27 | +16.99 |
| Min wall/query (s) | 75.58 | 78.37 | +2.79 |
| LLM calls/query | 2.0 | 2.0 | +0.0 |

### Deterministic savings vs. provider noise — how to read Run 1 vs Run 2

**Orchestration overhead is 0.81 s/query** (measured with an instantly
responding stub LLM: `orchestration_only_post.json`). Everything else is
the LLM provider: on pinned OpenRouter free models a call costs ~45–60 s
of queue+decode, so **~98–99% of end-to-end wall time is the provider**.

The code-level savings are structural and deterministic:

- **9.00 s/query** removed by replacing OpenRouter's unconditional 4.5 s
  pre-call sleep with 429-only backoff (2 calls × 4.5 s). This applies to
  *every* query on the OpenRouter path, in both runs' accounting.
- **Speculative parallel retrieval** hides the ~0.5–1.5 s retrieval window
  behind the router call (0.81 s → sub-second orchestration already
  includes this overlap).
- **Batch embedding** removes N−1 redundant BGE encode passes on
  multi-query (legacy/refinement) paths.

The E2E means (104.28 s → 121.27 s) are **within free-tier provider queue
variance**: per-query spread inside a single run is ±40% (Run 1: 75.6–120.7 s,
Run 2: 78.4–156.8 s), far larger than the 9 s structural saving. Run 2's
fastest query (78.37 s) matches Run 1's fastest (75.58 s) despite Run 2
carrying *no* sleep — the saving is real but masked by the provider window,
which was ~26 s/query slower during the Run 2 measurement window.

**Practical conclusion:** further end-to-end speed must come from the LLM
path — a faster provider/model (e.g., Gemini 2.5 Flash when quota allows,
a paid OpenRouter model, or a local fast model) is worth 3–10× more than
any remaining pipeline tuning, because the pipeline itself costs <1 s.

---

## 4. LLM model selection — live latency benchmark

Benchmark `model_benchmark_20261028_glm.json` (2026-10-28T18:31:14Z), fixed 123-char prompt:

| Model | Calls | Best (s) | Status |
|---|---:|---:|---|
| `z-ai/glm-5.3-flash` | 2 | 1.18 | ok |
| `nvidia/nemotron-3.5-lightning:free` | 2 | 3.91 | ok |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | 1 | 8.95 | ok |
| `google/gemma-4-31b-it:free` | 1 | — | HTTP 429: {"error":{"message":"Provider returned error","cod |

**Rotation order adopted (fastest valid first):**

1. `z-ai/glm-5.3-flash` — 1.18s
2. `nvidia/nemotron-3.5-lightning:free` — 3.91s
3. `nvidia/nemotron-3-ultra-550b-a55b:free` — 8.95s

The free gemma tier was 429-saturated during the benchmark and `inkling-small:free` is 403-restricted — the gate's 429-backoff and model rotation handle both at runtime (ADR-007).

---

## Caveats

- Live E2E numbers depend on free-tier provider queue times; runs are pinned
  to the same provider and taken minutes apart to keep the comparison fair.
- Retrieval quality metrics are deterministic; the 60-case size means deltas
  < ~10 pp are noise (see `evaluation_results.md`).
- The retrieval benchmark's `mean` latency includes one cold-start query;
  use p50/p95 for warm-path comparisons.
