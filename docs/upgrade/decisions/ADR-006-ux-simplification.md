# ADR-006 — UX simplification: fewer decisions, fewer ways to lose work

**Date:** 2026-10-28 · **Status:** Accepted

## Context

The sidebar exposed three decisions a user should never have to make, and two ways to destroy work with one click:

1. **fast vs hi_res parsing** — `fast` (pypdf, milliseconds) extracts zero figures/tables, so the product's headline feature (CLIP visual search, figure-grounded answers) silently did not work on it. Asking users to choose between "fast" and "correct" is decision fatigue with a degraded default.
2. **Reranking toggle** — the cross-encoder is measurably better and costs ~100 ms; there is no user-facing scenario where turning it off helps.
3. **Per-document delete + "Clear DB" buttons** — instant and irreversible, no confirmation. One mis-click destroyed hours of hi_res ingestion. And they were nearly pointless: stores are **per browser session** — reloading the tab already yields a fresh library, and idle sessions auto-purge after 2 hours.

Additionally the visual layer was rebuilt as **creamy light glassmorphism** (frosted panels over soft aurora gradients, `app/theme.py` design system, `.streamlit/config.toml` light theme) with sharply reduced on-screen text: one-line hero, 3 capability badges, 3 metric cards (Documents / Chunks / Figures), caption-free sidebar, two-line empty state.

## Decision

1. **hi_res is the only UI path** (`config.DEFAULT_PARSE_STRATEGY = "hi_res"`). The strategy selectbox is removed. `fast` remains available programmatically/CLI for huge text-only dumps (`parse_pdf(path, strategy="fast")`).
2. **Reranking is always on** in the UI. The toggle is removed; `config.RERANK_ENABLED` remains a programmatic kill-switch.
3. **Destructive buttons removed from the UI.** Per-doc delete and Clear DB are gone. In their place, one **"New Session"** button that rotates `session_id` and its storage paths — a fresh empty library and cleared chat where *nothing is deleted*: the old session directory is left for the 2-hour auto-purge. Backend cleanup functions (`delete_document_from_store`, `clear_all_documents_from_store`) remain for CLI/deliberate use.
4. Streamlit's top header is kept as a transparent layer (only the hamburger/deploy/decoration elements hidden) so the sidebar collapse/reopen arrows (» / «) stay reachable — hiding the whole header had made a collapsed sidebar impossible to reopen.

## Consequences

- The sidebar reads Upload → Library → New Session; no explanatory captions, no toggles.
- Every ingestion runs layout parsing: ingestion is slower, but the visual database is always populated and behavior is uniform.
- A bad ingest can no longer be fixed from the UI by deleting the document; the remedies are the New Session reset or the backend/CLI functions. Accepted, since the UI path for "start over" is non-destructive and the session model makes per-doc deletion rarely necessary.
- The design system is centralized in `app/theme.py`; visual retuning is a single-file change.
