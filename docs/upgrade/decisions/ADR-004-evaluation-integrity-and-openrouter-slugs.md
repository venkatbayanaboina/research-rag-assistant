# ADR-004 — Evaluation integrity fixes and OpenRouter slug refresh

**Date:** 2026-10-28 · **Status:** Accepted

## Part 1 — No fabricated judge scores (audit defect D4)

**Context:** `evaluation_suite/pipelines/run_features_audit.py` defaulted judge results to `"score": 96.0/98.0` whenever JSON parsing failed, systematically inflating the published "98.33% overall" features score.

**Decision:** judge failures now record `score: null` + `UNGRADED - judge failed: <reason>`. Averages are computed over graded entries only, with ungraded counts displayed. Historical reports produced under the old behavior are not trustworthy; re-run the pipeline to regenerate.

## Part 2 — README benchmark claim (audit defect D5)

The README's "Hybrid RAG (Dense + Sparse BM25)" matrix described a component absent from the code. The section now presents only reproducible numbers (the new retrieval benchmark) and marks the legacy tables as historical/unverifiable pending artifacts.

## Part 3 — OpenRouter free-model rotation was dead (new defect D15)

**Context:** found during the end-to-end smoke test: the hardcoded slugs `meta-llama/llama-3.2-3b-instruct:free` (404 — no longer free) and `openrouter/free` (intercepted by a safety classifier, returning `User Safety: safe`) made the entire OpenRouter fallback gate unusable even with a valid key. The live model list (2026-10-28, public `/api/v1/models`) shows a completely different free tier.

**Decision:** rotation defaults refreshed to verified-current slugs (`google/gemma-4-31b-it:free`, `nvidia/nemotron-3.5-lightning:free`, `google/gemma-4-26b-a4b-it:free`) and made overridable via `OPENROUTER_MODELS="slug1,slug2"` without code changes. Guard-model slugs (e.g. `nemotron-3.5-content-safety`) are excluded — they answer safety classification instead of the prompt.

**Consequence:** free slugs rotate frequently; when the fallback gate 404s again, set `OPENROUTER_MODELS` from the current list rather than editing code.
