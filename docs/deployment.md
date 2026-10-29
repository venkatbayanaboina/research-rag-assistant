# Deployment

## 1. Run locally (the supported path)

```bash
python3 main.py ui          # auto-uses the bundled rag/ venv
# or:
rag/bin/python3 -m streamlit run app/ui.py
```

Keys go in `.env` (copy `.env.example`). First query downloads model weights
(~2 GB) into `~/.cache/huggingface`.

## 2. Docker

```bash
docker build -t research-rag .
docker run -p 8501:8501 --env-file .env -v glm_cache:/root/.cache/huggingface research-rag
```

## 3. Streamlit Community Cloud (free public demo) — EXACT STEPS

1. Push this repository to GitHub (`git remote add origin … && git push -u origin main`).
2. share.streamlit.io → **Create app** → "Deploy a public app from GitHub" → pick the repo.
3. Main file path: `app/ui.py`. Python version: 3.12 (set under Advanced).
4. Add secrets (dashboard → Settings → Secrets), one per line:
   `OPENROUTER_API_KEY = "sk-or-…"`
   The gates read env vars, so Secrets work as-is. `packages.txt` (poppler-utils,
   libgl1) is already in the repo for the required apt packages.
5. Deploy → copy the app URL into `README.md` §Deployment below.

**Caveats:** cold start downloads ~2 GB of model weights (attach nothing; it caches
per container, and Community Cloud rebuilds containers — expect slow first answers
after redeploy). Large-PDF ingestion is CPU-bound. For a stable demo, use the
Docker image on a $5 VM instead.

**requirements note:** use `requirements.txt` on cloud (cross-platform);
`requirements.lock.txt` is the macOS reproducibility pin set and may contain
mac-only wheels.

> **Status:** scaffolding complete. **No public demo is deployed** — the link
> requires your GitHub / Streamlit Cloud accounts. Paste the URL here once live:
> **DEMO_URL: pending**

## 4. What the demo exposes

## 4. What the demo exposes

Upload → index (hi_res layout parsing) → chat Q&A with citation verification →
7-intent routing incl. evidence tables / contradiction analysis / gap exploration.
Session stores are isolated per visitor; nothing is shared between visitors.
