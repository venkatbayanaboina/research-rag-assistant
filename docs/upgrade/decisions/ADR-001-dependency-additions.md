# ADR-001 — Add rank_bm25 and langgraph as direct dependencies

**Date:** 2026-10-28 · **Status:** Accepted (installed into `rag/` venv, added to `requirements.txt`)

## Context

The upgrade requires (a) true sparse retrieval to pair with the existing dense BGE search, and (b) a stateful, inspectable orchestration layer. Neither existed in the repo (`repository_audit.md` §1.3).

## Decision

1. **`rank_bm25` (BM25Okapi)** for sparse lexical retrieval. Chosen over building a custom scorer because it is the de-facto reference implementation, dependency-free, and fast enough to build at query time over the current 4,722-chunk corpus (<1 s, cached by registry mtime/size).
2. **`langgraph`** for the agentic graph. Chosen over hand-rolled control flow because the spec mandates LangGraph, and its StateGraph gives typed state, conditional edges, and transition testability with no runtime service dependency. `langchain` core packages arrive transitively with `langgraph`; no LangChain wrappers were introduced into retrieval/embedding code.

Both were installed with `rag/bin/python3 -m pip install` (the venv's console-script shebangs are stale after a repo move — see `repository_audit.md` §1.4). No existing package versions were changed.

## Consequences

- Python 3.14 compatibility verified by import + full unit suite run (37 passed).
- BM25Okapi IDF is ≤ 0 when a term matches ≥ half the corpus; such chunks are excluded from sparse results (`score <= 0` cutoff). This is desirable for fusion (dense covers generic terms) and is pinned by tests.
- Free-tier LLM quotas (Gemini 20 req/day) can still exhaust; the graph degrades deterministically (verified in the end-to-end smoke test).
