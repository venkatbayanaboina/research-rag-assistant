# Security Review — Post-Upgrade

**Date:** 2026-10-28 · Scope: the multimodal RAG assistant after the agentic upgrade. Ratings: Low / Medium / High (impact × likelihood for this project's actual deployment: local single-user Streamlit, cloud LLM APIs).

## 1. Prompt injection via retrieved documents — **Low for this deployment (recalibrated 2026-10-28)**

PDF text is untrusted content and is concatenated into generation prompts (`answering.py`) without delimiting isolation. **Honest threat-model assessment:** for a single-user local research assistant whose corpus is voluntarily-curated academic papers, with a fixed-function agent (no tools, no shell, no network egress beyond the chosen LLM provider) and a human reading every answer, the realistic attacker set is empty and the blast radius is bounded to a briefly-misleading answer about a paper the user owns. Injection becomes a genuine High risk only if the project gains: web-connected retrieval, tool/plugins, multi-tenancy, or automated consumption of outputs — at that point, implement the layered plan sketched below (delimiting, canary tokens, injection-corpus ASR benchmark).

**Mitigations already in place:** citation verification flags uncited claims (injected "don't cite" instructions trip the detector — measured: 0 fabricated citations across 60 evaluated answers); research-action extraction is page-anchored with provenance validation (fabricated references dropped).
**Cheap hygiene worth doing regardless (~30–60 min, $0):** wrap retrieved chunks in `<untrusted_document>` tags + one instruction-hierarchy rule in all generation prompts; a handful of unit tests asserting the delimiters exist. Deferred by choice, not forgotten.

## 2. Secrets handling — **Low**

- `.env` is gitignored and untracked (verified via `git ls-files`); `.env.example` contains placeholders only.
- Gates load `.env` per call (`load_dotenv(..., override=True)`) — a local-file read, not a network exposure.
- Console logs print error bodies from provider responses; provider error payloads do not contain keys, and request headers (which carry the bearer token) are never logged. No secret-logging path found.
- **Keep:** never paste `.env` contents into issue trackers; the OpenRouter HTTP-Referer header intentionally identifies the public repo.

## 3. Local subprocess side effects — **Medium**

`OllamaGate.ensure_ollama_running` spawns `ollama serve` on the host when the fallback chain reaches it (observed in the smoke test). It fails cleanly when Ollama is absent, but a library call starting servers is surprising.

**Recommended next:** gate behind an explicit `ENABLE_OLLAMA_AUTOSTART` config flag (default off). *Not done here to keep the change set focused; flagged for the next pass.*

## 4. Untrusted-file ingestion — **Medium (pre-existing, unchanged)**

`unstructured`/`pypdf` parse arbitrary uploaded PDFs. Parser vulnerabilities are out of this project's control; exposure is bounded because parsing happens in the user's own process with their own privileges and files stay under `storage/`. Streamlit's default upload size limits apply.

## 5. Session isolation — **Medium (improved by audit fixes, one gap remains)**

- Storage isolation per Streamlit session is real (`storage/sessions/<uuid>/`).
- Fixed in this upgrade: figure selection now reads the session registry (D3) instead of the global one.
- Fixed in this upgrade: per-document delete and "Clear DB" buttons were removed from the UI (ADR-006). They were instant and irreversible with no confirmation — a mis-click destroyed hours of hi_res ingestion. The replacement **New Session** control rotates to a fresh store *without deleting anything* (old data waits for the 2-hour auto-purge), so the UI no longer offers a one-click data-loss path. Backend cleanup functions remain for deliberate CLI use.
- Remaining gap: `sidebar.PROGRESS_STATE` is process-global (D8) — concurrent users see/cancel each other's ingestion progress. No data corruption path found (each session's stores are separate); it is a UX/cross-session-information issue. Scheduled as the next reliability fix.

## 6. Concurrency correctness — **Medium (improved)**

`db_lock` is now reentrant and covers the full read-embed-write sequence in `add_document_to_store` (D6), removing the lost-update window on the registry/index pair. Single-process assumption stands (see limitations).

## 7. Network egress — **Low**

Outbound calls: Gemini, OpenRouter, Hugging Face (model loads), fonts.googleapis.com (UI CSS), Ollama (localhost only). No user-document content is sent anywhere except the configured LLM providers — which is the product's purpose. The OpenRouter payload includes base64 answer figures; users should assume uploaded documents reach the provider they configure.

## 8. New attack surface introduced by this upgrade — **Low**

- The verification Layer-2 call sends claim/evidence snippets to the LLM provider — same trust boundary as the main generation call; no new egress class.
- No new user input is trusted: `[Source N]` markers are parsed from *model output* only, and malformed markers can only affect the verification status shown to the user (worst case: a wrong warning icon). No eval/exec sinks exist in any new module.
- `langgraph` state is in-process; no serialization of untrusted data.

## 9. Summary of security-relevant changes in this upgrade

| Change | Security effect |
|---|---|
| Citation verification + verified/qualified/failed chips | Detects ungrounded answers (incl. injection-influenced ones) |
| Session-aware image registry (D3) | Removes cross-session information bleed |
| Reentrant full-scope DB lock (D6) | Removes registry/index race window |
| Honest judge scoring (D4) | Prevents silently inflated quality reporting |
| OPENROUTER_MODELS override | Lets operators rotate models without code edits |
| Destructive UI controls removed; non-destructive New Session (ADR-006) | Eliminates one-click irreversible data loss from the UI |
| hi_res-only ingestion (ADR-006) | Uniform behavior; `fast` remains a code/CLI path, not a silent degradation |
| Visual retrieval evidence rules (ADR-005) | Images attach only on verifiable evidence — fewer unvetted visuals reach the model |
