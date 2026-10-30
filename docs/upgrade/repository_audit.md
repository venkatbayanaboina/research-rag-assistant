# Repository Audit — research-rag-assistant

**Date:** 2026-10-28 · **Scope:** Phase 0 discovery for the "Agentic Multimodal Research Intelligence" upgrade. Every claim below was verified against the actual source on `main` (commit `38f3978` + uncommitted working-tree changes). No application code was modified during this audit.

---

## 1. Verified Architecture (as-built)

### 1.1 Modules and responsibilities

| Module | Lines | Verified role |
|---|---|---|
| `main.py` | 184 | CLI entry point: `ingest`, `query`, `summarize`, `compare`, `chat`, `ui` subcommands |
| `app/ui.py` | 96 | Streamlit entry; per-session storage isolation under `storage/sessions/<uuid>/`; 2-hour session cleanup |
| `app/components/sidebar.py` | 208 | Upload, parsing-strategy select, background ingestion thread, progress fragment, delete/clear controls |
| `app/components/chat.py` | 154 | Chat loop, invokes `execute_rag_pipeline`, renders sources/figures expanders |
| `src/core/ingestion.py` | 100 | `parse_pdf(file, strategy)`: `fast` = pypdf text + heuristic title detection; `hi_res` = `unstructured.partition_pdf` with YOLOX + Image/Table cropping |
| `src/core/chunker.py` | 182 | `chunk_by_title` grouping (2000/1500/100/200 char params), section-title tracking, caption association for images/tables (±8 element window, position/type heuristics) |
| `src/core/embedder.py` | 117 | Lazy singleton loaders: BGE `BAAI/bge-large-en-v1.5` (1024-d, query prefix), CLIP `clip-ViT-B-32` (512-d); CUDA→MPS→CPU device pick |
| `src/core/vector_store.py` | 485 | Dual FAISS `IndexFlatIP` (text 1024-d, image 512-d), JSON registries, dense search, **visual hybrid search** (CLIP dense ≥0.28 + caption keyword sparse fused with +0.05 boost), delete-with-reconstruction, session-path resolution |
| `src/core/reranker.py` | 78 | `cross-encoder/ms-marco-MiniLM-L-6-v2` rerank of top-15 → top-5; `torch.set_num_threads(1)` |
| `src/core/generator/router.py` | 70 | LLM intent classification: `STANDARD_CHAT / SUMMARY / SECTION_SUMMARY / COMPARISON` + `target_docs` resolution; JSON mode, temp 0, falls back to `STANDARD_CHAT` on any error |
| `src/core/generator/orchestrator.py` | 225 | `execute_rag_pipeline`: intent dispatch; standard path does **query expansion (5 LLM alternatives)** → per-query dense retrieval → **RRF fusion (k=60)** over text and images → top-5 text / top-2 images → `generate_answer` |
| `src/core/generator/answering.py` | 69 | Single Gemini call with `[Source N - file (Page X)]`-labeled context, grounding rules, LaTeX formatting rules, PIL images attached |
| `src/core/generator/summarization.py` | 187 | Direct single-pass (≤60k chars) vs. hierarchical map-reduce summarization; key-figure selection by caption keywords |
| `src/core/generator/comparison.py` | 73 | N-way (2–5+) comparison table generation, all-paper context in one prompt |
| `src/core/generator/gates/` | ~400 | `BaseModelGate` + `GeminiGate` (retry on 429/503, daily-quota detection) → `OpenRouterGate` (4.5 s throttle, free-model rotation, JSON validation) → `OllamaGate` (auto-starts local server, model auto-pick). Permanent switch on failover |
| `src/core/utils/profiler.py` | 35 | JSONL step timer → `storage/performance_logs.jsonl` |
| `config.py` | 42 | Hardcoded model names, chunk/rerank constants, storage paths |
| `tests/` | ~900 | `test_features.py` (3 prompt-construction mocks), `test_rag_eval.py` (DeepEval faithfulness/relevancy), `run_eval.py`, `generate_test_cases.py`, `evaluate_layout_parsers.py`, `analyze_performance.py` |

### 1.2 Actual request flow (traced end-to-end)

```
Streamlit chat input (app/components/chat.py:61)
  → session lock via st.session_state.thinking
  → execute_rag_pipeline(prompt, indexed_docs, use_rerank, chat_history)   [orchestrator.py:49]
      → route_user_intent (LLM call #1, JSON mode)
      ├─ SUMMARY / SECTION_SUMMARY / COMPARISON → direct registry reads → LLM generation (no retrieval)
      └─ STANDARD_CHAT:
           → generate_similar_queries (LLM call #2: 5 rewrites)
           → for each of 6 queries: search_store (BGE dense, top-15 → cross-encoder top-5)   [LLM call count: 0]
           → RRF k=60 fusion over text rankings → top-5 chunks
           → for each of 6 queries: search_image_store (CLIP dense + caption keyword sparse)
           → RRF over image rankings (+2.0 boost for explicit "fig/table N" pattern hits on the original query) → top-2
           → generate_answer (LLM call #3: context + images, temp 0)
  → assistant message + sources/figures stored in st.session_state.chat_history
```

**Source of truth for provenance:** `storage/chunks.json` (registry; `chunk_id` = FAISS row order) and `storage/images_registry.json` (`image_id` = FAISS row order). Page numbers come from `unstructured` element metadata at parse time. Citations in answers are **not** validated against these registries anywhere in the current code.

### 1.3 Expected-vs-verified capabilities (spec §Phase 0)

| Expected capability | Verified? | Notes |
|---|---|---|
| Layout-aware PDF ingestion | ✅ | `unstructured` + YOLOX (`hi_res`), pypdf fast path |
| YOLOX + OCR-based processing | ⚠️ Partial | YOLOX layout model used; **no OCR pass** (`ocr_languages` not configured); relies on native text layer |
| BGE-Large text embeddings | ✅ | `BAAI/bge-large-en-v1.5`, query-prefix instruction tuning |
| CLIP visual embeddings | ✅ | `clip-ViT-B-32` via sentence-transformers |
| Separate FAISS indexes | ✅ | 1024-d text + 512-d image `IndexFlatIP` |
| Dense retrieval | ✅ | FlatIP inner product on normalized vectors |
| **BM25 retrieval** | ❌ **Absent** | No `rank_bm25`/BM25 implementation anywhere in `src/`; `requirements.txt` has no BM25 package |
| Reciprocal Rank Fusion | ✅ (dense-only) | RRF over multi-query *dense* rankings in the orchestrator; visual search fuses CLIP-dense with keyword scores by max+0.05, not RRF |
| Cross-Encoder reranking | ✅ | ms-marco-MiniLM-L-6-v2, toggleable |
| LLM intent routing | ✅ | Gemini JSON router with 4 intents |
| Multi-provider fallback | ✅ | Gemini → OpenRouter → Ollama, permanent switch, quota-aware |
| Streamlit UI | ✅ | Session-isolated stores, background ingestion thread |
| DeepEval evaluation | ✅ | `tests/test_rag_eval.py` + `evaluation_suite/pipelines/` |
| **LangGraph / LangChain** | ❌ Absent | Not in requirements, not imported |

### 1.4 External environment (verified)

- Runtime venv: `rag/` (**Python 3.14.4**) — torch 2.12.0, faiss-cpu 1.14.2, sentence-transformers 5.5.1, unstructured, deepeval 4.0.7, pytest 9.1.1, streamlit 1.57.0, google-genai.
- `rag/` venv is **broken for direct script invocation**: entry-point shebangs point to the old absolute path (`/Users/nanibayanaboina2750/Desktop/research-rag-assistant/...`) because the repo folder was moved. `rag/bin/python3 -m pip` works.
- System `python3` (3.9.6) cannot run the app (no faiss/sentence-transformers/streamlit).
- Model weights cached in `~/.cache/huggingface/hub` (BGE-large, CLIP, MiniLM cross-encoder, YOLOX layout) — local inference is feasible offline.
- `rank_bm25` and `langgraph` were **not installed** at audit time; installed into `rag/` venv during Phase 1 setup (recorded in `decisions/ADR-001`).

---

## 2. Known Defects (verified by reading the code; each cites file:line)

### D1 — CLI `compare` crashes (broken call signature) — **High**
`main.py:84` calls `generate_comparison(doc_a_name, doc_a_chunks, doc_b_name, doc_b_chunks)` (4 positional args), but the current signature is `generate_comparison(docs_data)` where `docs_data` is a list of `{"name", "chunks"}` dicts (`comparison.py:11`). The 4-arg call makes `docs_data` a string; iteration hits `doc["name"]` → `TypeError`. The `main.py compare` subcommand is unusable.

### D2 — Unit test asserts the same broken signature — **High**
`tests/test_features.py::test_generate_comparison_mock` (line 89) calls `generate_comparison` with the old 4-arg form and therefore fails against current code. The test was not updated when the comparison engine was generalized (commit `e8d8bfa`).

### D3 — Session-isolation leak in key-figure selection — **Medium**
`summarization.get_key_images_for_doc` (line 17) reads `config.IMAGES_REGISTRY_PATH` directly instead of the session-aware `vector_store.get_paths()`. Inside Streamlit, summaries attach figures from the **global** image registry, not the user's session store (wrong/missing figures; cross-session data bleed).

### D4 — Evaluation scores are fabricated on JSON-parse failure — **High (evaluation integrity)**
`evaluation_suite/pipelines/run_features_audit.py` defaults judge output to `"score": 96.0` / `98.0` when JSON parsing fails (lines 113, 146). Any parse failure is recorded as a near-perfect score, systematically inflating the published report (`deepeval_multimodal_features_report.md`, README §5.C "98.33% overall").

### D5 — README benchmark table references a component that does not exist — **High (documentation integrity)**
README §5.A reports "Hybrid RAG *(Dense + Sparse BM25)*" results (31.10%→48.57% table). No BM25 implementation exists in the repository. Those numbers are not reproducible from this codebase as-is.

### D6 — `add_document_to_store` performs read-modify-write without holding `db_lock` — **Medium**
`vector_store.py:85-181`: only `delete_document_from_store` (called first) takes the lock; the subsequent registry read, FAISS add, and JSON writes run unlocked. Two concurrent ingests can interleave and lose chunks or corrupt row↔`chunk_id` alignment. (The README's Challenge-3 narrative claims this was audited; the lock boundary is still incomplete.)

### D7 — Image-ID misalignment when image files disappear — **Medium**
`embed_image_batch` silently skips missing files (`embedder.py:93`), but `add_document_to_store` assigns `image_id = start_id + idx` over **all** chunks (`vector_store.py:170-173`). If any file is missing between chunking and embedding, FAISS rows and registry IDs shift → wrong image returned for an ID. Also `run_eval.py` explicitly resets model caches mid-process — evidence of this class of fragility.

### D8 — Module-global `PROGRESS_STATE` shared across all Streamlit sessions — **Medium**
`sidebar.py:14-21`: ingestion progress state is process-global. Two concurrent users' uploads overwrite each other's status, and either can cancel the other's ingestion.

### D9 — Ollama gate spawns subprocesses as a side effect — **Low/Medium**
`ollama.py:30` runs `subprocess.Popen(["ollama", "serve"])` from inside request handling. Surprising side effect from a library call; unbounded lifetime; no opt-out.

### D10 — Broad bare `except:` blocks — **Low**
`vector_store.py:34, 473-474, 482-484`, `summarization.py:24, 90`, `sidebar.py:44-46`, `ui.py:45-46` swallow all exceptions (including `KeyboardInterrupt`/`SystemExit`), masking real failures.

### D11 — No token-usage or cost tracking — **Low**
Gates return only text; `log_timing` records durations only. No token counts are captured anywhere, so cost is unmeasurable today.

### D12 — Router/JSON parsing without schema validation — **Low**
`router.py:60` and `orchestrator.py:40` do `json.loads` + ad-hoc key checks; malformed-but-valid JSON (e.g. wrong intent enum) passes through silently.

### D13 — Rag `requirements.txt` is incomplete — **Low**
Missing direct deps that are imported: `pypdf`, `torch`, `requests`, `deepeval`, `pytest`, `rank_bm25` (after this upgrade). Works only because transitive installs happen to provide them.

### D14 — Old chunks lack `section_title` — **Low**
`storage/chunks.json` mixes schemas: older chunks have no `section_title` key (verified: 4,722 chunks, 9 docs). Any code assuming the key must use `.get()`.

---

## 3. Missing Tests

- **Zero tests** for: chunker, ingestion, embedder, vector_store (add/search/delete/fusion), reranker, router fallback behavior, orchestrator RRF math, gates failover, UI components.
- `test_rag_eval.py` requires live LLM access and indexed docs — it is an integration test, not a unit test, and cannot run in CI without secrets.
- No regression test pins the `generate_comparison(docs_data)` signature (the defect that caused D1/D2).

## 4. Architectural Risks

1. **Single-process globals** (`db_lock`, `_active_gate_instance`, `PROGRESS_STATE`, model singletons) — correct only for one Streamlit worker; unsafe under `streamlit run` with multiple threads or any WSGI deployment.
2. **LLM-in-the-loop routing without fallback depth** — if routing fails, the query silently degrades to `STANDARD_CHAT` (may still be fine, but is unobservable).
3. **Prompt-injection surface** — retrieved PDF text is concatenated into generation prompts with no delimiting/escaping policy; a malicious PDF can override generation rules (spec Phase 9). No tool execution exists today, so impact is bounded to answer content.
4. **Index/registry coupling** — FAISS row order ↔ registry list order invariant is maintained only by convention (and is violated by D7). No checksum or rebuild-verification exists.
5. **Git hygiene** — runtime artifacts (`storage/chunks.json`, `*.faiss`, `extracted_images/`, `performance_logs.jsonl`) are tracked in git; every ingestion dirties the working tree (visible in current `git status`). `.env` is correctly gitignored and untracked.
6. **Repo layout** — a Python 3.14 venv lives inside the repo at `rag/` (gitignored) with absolute-path shebangs that break when the folder moves; `archives/` and `development_phases/` contain historical copies that confuse discovery.

## 5. Improvement Opportunities (feeding the implementation plan)

Ranked by expected impact on answer quality and reliability:

1. **True hybrid retrieval (BM25 + dense + RRF)** on the text side — exact lexical matches (equations names, dataset names, "Table 3") are the dense model's known weakness; also makes the README's "Hybrid RAG" claim true. *Highest quality-per-effort.*
2. **LangGraph orchestration** around the existing components: explicit state, bounded refinement loop, evidence-sufficiency gate, observable transitions — replacing the implicit control flow in `orchestrator.py`.
3. **Citation verification** (deterministic Layer 1 + optional LLM Layer 2) — currently nothing checks that an answer's claims map to retrieved sources.
4. **Fix D1–D6** — broken CLI path, broken test, session leak, fabricated scores, false README claim.
5. **Retrieval benchmark harness** — `evaluation_suite/dataset/gold_qa_dataset.json` (315 papers, QA pairs with evidence page/section) enables a reproducible Recall@k / MRR benchmark on the 4 indexed arXiv papers without any LLM judge.
6. Token/cost accounting in the gates (cheap, enables Phase 7 performance metrics).
7. Lock-scope fix (D6) and per-session progress state (D8).
8. De-duplicate `archives/` from discovery and ignore runtime storage artifacts.

---

## 6. What was NOT changed during this audit

No application code was modified in Phase 0. Phase 1 additions were limited to: installing `rank_bm25` + `langgraph` into the existing `rag/` venv (no version changes to existing packages) and creating the `docs/upgrade/` documentation set. See `baseline.md` for measured facts vs. assumptions.

---

## 7. Addendum — second-wave findings (2026-10-28, post-implementation)

Defects discovered after this audit was written, during implementation and live testing. All are fixed; details live in the referenced ADRs.

| ID | Finding | Fix |
|---|---|---|
| D15 | OpenRouter free-model rotation dead: hardcoded slugs return 404 / safety-classifier interception; fallback gate unusable even with a valid key | Refreshed rotation to live-verified slugs; `OPENROUTER_MODELS` env override (ADR-004) |
| D16 | Visual search mixed incompatible score spaces (CLIP cosine vs. keyword scores) behind a magic 0.28 threshold; substring token matching let the digit "1" in "Show me Figure 1" match "P1"/"12" caption text; single ubiquitous tokens ("model") qualified images; hardcoded `[:2]` padded every answer to 2 images — measured: 2 wrong tables returned for a non-existent figure | Evidence-rule rewrite (`visual_ranking.py`): relative dense calibration, IDF-weighted token-SET matching, explicit-reference patterns, dense∩sparse agreement, k as cap. A/B-verified (ADR-005) |
| D17 | UI decision fatigue + one-click data loss: fast/hi_res picker (fast silently disables the multimodal feature set), reranking toggle, and irreversible per-doc delete / Clear DB buttons with no confirmation | hi_res-only ingestion, reranking always on, destructive controls replaced by a non-destructive New Session (store rotation + 2h auto-purge) (ADR-006) |
| D18 | Streamlit header hidden wholesale by the UI theme → a collapsed sidebar could not be reopened (the » arrow lives in the header) | Header kept as a transparent layer; only menu/deploy/decoration elements hidden |
| D19 | CLI failed on system Python (`python3 main.py ui` → ModuleNotFoundError: streamlit) | `main.py ensure_runtime()` re-executes the CLI through the bundled venv when deps are missing |
| D20 | `rag/` venv console-script shebangs pointed to a pre-move absolute path (`rag/bin/pip` broken) | Documented workaround (`rag/bin/python3 -m pip`); CLI self-heal (D19) covers app entry |
