# ADR-005 — Visual retrieval: evidence rules instead of fixed thresholds

**Date:** 2026-10-28 · **Status:** Accepted (A/B-verified on a hi_res-ingested 21-page, 21-figure PDF)

## Context

The pre-existing visual search mixed incompatible score spaces — CLIP cosine similarity against keyword scores (0.45 explicit / 0.28+ token-ratio) — behind a magic 0.28 absolute threshold, matched tokens by *substring* against caption text (so the digit "1" in "Show me Figure 1" matched "P1", "12"…), counted filename tokens as evidence, and hard-coded `[:2]`, so every answer was padded to exactly 2 images whether or not any were relevant.

**Measured defects** (baseline captured on `Social-Engineering-Attack.pdf`, hi_res-ingested: 21 visual chunks):

| Query | Old behavior |
|---|---|
| "Show me Figure 1" (no Figure 1 exists in the doc) | 2 wrong tables (score 0.4 via stray-digit matches) |
| "table showing CMRR differential gain" | correct table **+ 1 irrelevant noise table** |

## Decision

New pure module `src/core/visual_ranking.py` + rewritten `vector_store.search_image_store`:

1. **Relative dense calibration** — keep CLIP hits within `IMAGE_DENSE_RELATIVE_MARGIN` (0.06) of the best hit *for this query*, above an absolute noise floor (0.20). Absolute cutoffs fail because CLIP's similarity scale shifts per query/domain.
2. **IDF-weighted sparse matching** — caption document frequencies are computed per registry; a token's evidence is `ln(N/(1+df))`. Ubiquitous tokens ("model", "results") carry ≈0 evidence; single ubiquitous tokens can no longer qualify an image. Token matching uses caption **token sets**, not substrings. Structural nouns ("figure", "table", "diagram") and bare digits are excluded from topical evidence (they only matter inside explicit-reference patterns).
3. **Explicit references** ("figure 3", "table ii", subfigure letters) still match captions by pattern and always qualify.
4. **Evidence-based qualification** — an image attaches to an answer only if: explicit reference, **agreement** (found independently by both dense and sparse), or strong caption evidence (IDF sum ≥ `IMAGE_SPARSE_STRONG_IDF` = 2.5). Dense-only hits are rejected.
5. **k is a cap, not a quota** — `config.K_IMAGE_RETRIEVAL = 2` is a maximum; zero is a valid result. Both the agentic and legacy pipelines consume the cap instead of a hardcoded slice; the agent's original-query boost now keys on the `explicit` flag rather than a magic score value.

## Measured result (same store, same queries, new logic)

| Query | Old | New |
|---|---|---|
| "Show me Figure 1" | 2 wrong tables | **0** (correct — no Figure 1 exists) |
| "table showing CMRR differential gain" | right table + noise | **exactly the right table** |
| "chatgpt interface screenshot" | 2 relevant | 2 relevant (retained via dense∩sparse agreement) |
| Generic / text-only questions | 0 | 0 |

Precision strictly improved; genuinely visual queries unchanged. Rules are pinned by 12 unit tests (`tests/test_visual_ranking.py`) covering the exact failure modes above.

**Tuning:** `IMAGE_DENSE_RELATIVE_MARGIN`, `IMAGE_DENSE_ABS_FLOOR`, `IMAGE_SPARSE_MIN_IDF` (1.2), `IMAGE_SPARSE_STRONG_IDF` (2.5), `K_IMAGE_RETRIEVAL` — all in `config.py`.

**Limitation:** qualification quality depends on caption/OCR quality from `hi_res` parsing; a figure with an empty caption can only be found by the dense channel in agreement with *some* sparse evidence, which by design it will not have. Low-caption-quality corpora will therefore attach fewer images — deliberately conservative.
