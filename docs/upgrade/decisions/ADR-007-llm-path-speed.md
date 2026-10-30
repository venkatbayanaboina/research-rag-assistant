# ADR-007 — LLM-path speed: measured model rotation and the provider ceiling

**Date:** 2026-10-28 · **Status:** Accepted

## Context

After the pipeline optimizations (ADR-006-era sleep removal, speculative retrieval, batch embedding), measured data showed **~98–99% of end-to-end wall time is the LLM provider**: orchestration overhead is 0.81 s/query with a stubbed LLM, while a pinned OpenRouter free-model call costs ~45–60 s of queue+decode. Further pipeline tuning cannot move end-to-end latency; the lever is the model path itself.

## Decision

1. **Rotation ordered by live measurement, not assumption.** A micro-benchmark (`benchmark_openrouter_models.py`) sends a fixed prompt to each candidate free model and records per-call latency and validity. Measured 2026-10-28: the free gemma tier was **429-saturated**, `inkling-small:free` returns 403 (restricted), and the nemotron models served requests (lightning 4.3 s best / 18.9 s confirm; ultra-550b 8.6–14.3 s). New default rotation:
   1. `nvidia/nemotron-3.5-lightning:free` (fastest valid)
   2. `nvidia/nemotron-3-ultra-550b-a55b:free`
   3. `google/gemma-4-31b-it:free` (fast earlier in the day; saturates under load — kept as later fallback)
   4. `google/gemma-4-26b-a4b-it:free`
   The existing 429-only backoff plus model rotation on failure makes the gate resilient to per-model saturation.
2. **Re-benchmarkable:** the benchmark script is part of the suite; when slugs rot or saturate, re-run it and update the rotation (or set `OPENROUTER_MODELS`).

## What this does and does not buy

- Avoiding a saturated head model saves its 429 round-trips (measured: 0.4–0.9 s per rejection before rotating).
- It does **not** change the free-tier queue ceiling: measured per-call latency on working free models is 4–19 s with high variance. The deterministic savings already banked (9 s/query sleep removal) remain the main code-level win.

## Consequence / next lever

End-to-end latency is now governed by provider choice: Gemini 2.5 Flash (fast, 20 req/day free), a paid OpenRouter model, or a fast local model would each cut the ~45–60 s/call to 1–5 s — a 10–30× end-to-end improvement no pipeline change can match. Token **streaming** to the UI is the other high-value next step (perceived latency drops to first-token time). Both are roadmap items, not implemented here.

## Addendum (2026-10-28, later) — GLM-5.3-Flash promoted to rotation head

Live verification of `z-ai/glm-5.3-flash` on OpenRouter ($0.15/M in, $0.50/M out,
1 M context, modality `text+image+video→text`): plain call **1.18–1.25 s**, JSON-mode
router call **1.95 s with valid parsed JSON**, image-attached call **3.30 s** — versus
3.9–4.6 s for the free nemotron head (whose 2026-10-28 benchmark JSON shows the gemma
tier still 429-saturated). New default rotation:

1. `z-ai/glm-5.3-flash` (paid, ~$0.001/call at our prompt sizes)
2. `nvidia/nemotron-3.5-lightning:free`
3. `nvidia/nemotron-3-ultra-550b-a55b:free`
4. `google/gemma-4-31b-it:free`

Cost impact on the planned full evaluation: judge + generation on GLM-5.3-Flash
≈ **$0.30–0.50 total**, so the owner's $1 budget covers a full pass with margin
(previously $0.65–1.10 on Gemini). End-to-end expectation: LLM call latency drops
from ~45–60 s (free tier) to ~2–5 s — a ~10–20× end-to-end improvement, confirming
the ADR's thesis that the provider was the bottleneck.
