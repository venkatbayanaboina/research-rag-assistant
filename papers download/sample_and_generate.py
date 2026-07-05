"""
sample_and_generate.py
-----------------------
1. Randomly samples 100 questions from rag_retrieved_answers.json
2. Generates RAG answers using Cerebras API (gpt-oss-120b)
3. Saves results to papers download/sample_100_answers.json
"""

import os
import sys
import json
import time
import random
import requests
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAG_INPUT  = ROOT / "papers download" / "rag_retrieved_answers.json"
SAMPLE_OUT = ROOT / "papers download" / "sample_100_answers.json"

# ── Cerebras API ─────────────────────────────────────────────────────────────
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "YOUR_CEREBRAS_API_KEY_HERE")
API_URL  = "https://api.cerebras.ai/v1/chat/completions"
MODEL    = "gpt-oss-120b"
RATE_LIMIT_DELAY = 6.0   # seconds between requests
MAX_CHUNKS = 3            # top-N retrieved chunks to use as context
SAMPLE_SIZE = 100
RANDOM_SEED = 42

def load_full_data() -> dict:
    with open(RAG_INPUT) as f:
        return json.load(f)

def load_sample() -> dict:
    if SAMPLE_OUT.exists():
        with open(SAMPLE_OUT) as f:
            return json.load(f)
    return {}

def save_sample(data: dict):
    temp = SAMPLE_OUT.with_suffix(".tmp")
    with open(temp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(temp, SAMPLE_OUT)

def query_cerebras(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 512,
    }
    backoff = 5
    for attempt in range(6):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            elif r.status_code == 429:
                print(f"  ⚠️  Rate limit. Backing off {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
            else:
                print(f"  ⚠️  HTTP {r.status_code}: {r.text[:100]}")
                time.sleep(3)
        except Exception as e:
            print(f"  ⚠️  Exception: {e}")
            time.sleep(3)
    return "[ERROR: Cerebras failed after retries]"

def build_prompt(question: str, chunks: list) -> str:
    context_blocks = []
    for j, chunk in enumerate(chunks[:MAX_CHUNKS], 1):
        context_blocks.append(
            f"[{j}] Source: {chunk.get('source_file','')}, Page: {chunk.get('page','')}\n"
            f"Text: {chunk.get('text','')}"
        )
    context_str = "\n\n".join(context_blocks)
    return (
        "You are a helpful research assistant. Answer the following question based ONLY on the provided context.\n"
        "RULES:\n"
        "1. Use ONLY the provided context. Do not use your own knowledge.\n"
        "2. Output ONLY the direct answer. No introduction, no prefix.\n"
        "3. If the context is insufficient, say so clearly.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

def main():
    print("=" * 60)
    print("📋  Step 1: Sample & Generate (Cerebras gpt-oss-120b)")
    print("=" * 60)

    full_data = load_full_data()
    sample_data = load_sample()

    # Pick or reload the same 100 question IDs
    if sample_data:
        sample_qids = list(sample_data.keys())
        print(f"✅  Loaded existing sample of {len(sample_qids)} questions.")
    else:
        random.seed(RANDOM_SEED)
        all_qids = list(full_data.keys())
        sample_qids = random.sample(all_qids, SAMPLE_SIZE)
        # Initialise sample_data with metadata
        for qid in sample_qids:
            item = full_data[qid]
            sample_data[qid] = {
                "question_id":     item["question_id"],
                "paper_title":     item.get("paper_title", ""),
                "question_type":   item.get("question_type", ""),
                "difficulty":      item.get("difficulty", ""),
                "question":        item["question"],
                "expected_answer": item.get("expected_answer", ""),
                "evidence":        item.get("evidence", {}),
                "retrieved_chunks": item.get("retrieved_chunks", []),
                "rag_answer":      "",
                "latency_sec":     0.0,
            }
        save_sample(sample_data)
        print(f"🎲  Sampled {SAMPLE_SIZE} random questions (seed={RANDOM_SEED}).")

    # Find pending entries
    pending = [qid for qid in sample_qids
               if not sample_data[qid].get("rag_answer","").strip()
               or sample_data[qid].get("rag_answer","").startswith("[ERROR")]
    print(f"⏳  Pending: {len(pending)} / {len(sample_qids)}")
    print(f"⏱️   Estimated time: ~{len(pending) * RATE_LIMIT_DELAY / 60:.1f} minutes")
    print()

    if not pending:
        print("✅  All sample questions already answered!")
        print(f"📄  Results saved to: {SAMPLE_OUT}")
        return

    for idx, qid in enumerate(pending, 1):
        item = sample_data[qid]
        prompt = build_prompt(item["question"], item["retrieved_chunks"])

        t0 = time.time()
        ans = query_cerebras(prompt)
        latency = time.time() - t0

        sample_data[qid]["rag_answer"]  = ans
        sample_data[qid]["latency_sec"] = latency
        save_sample(sample_data)

        status = "✓" if not ans.startswith("[ERROR") else "✗"
        print(f"  {status} [{idx}/{len(pending)}] {qid} — {latency:.1f}s | len={len(ans)}")

        time.sleep(RATE_LIMIT_DELAY)

    print()
    print("🎉  Generation complete!")
    print(f"📄  Saved to: {SAMPLE_OUT}")

if __name__ == "__main__":
    main()
