import os
import sys
import json
import time
import requests
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAG_OUTPUT = ROOT / "papers download" / "rag_retrieved_answers.json"

# ── Cerebras API Settings ─────────────────────────────────────────────────────
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "YOUR_CEREBRAS_API_KEY_HERE")
API_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL = "gpt-oss-120b"

# Cerebras free tier — target ~10 RPM (1 req/6s) to avoid TPM limits
RATE_LIMIT_DELAY = 6.0
# Max context chunks to send per request (reduces token usage)
MAX_CHUNKS = 2

def load_data() -> dict:
    if not RAG_OUTPUT.exists():
        print(f"❌ {RAG_OUTPUT} not found.")
        sys.exit(1)
    with open(RAG_OUTPUT) as f:
        return json.load(f)

def save_data_atomic(data: dict):
    temp_path = RAG_OUTPUT.with_suffix(".tmp")
    try:
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, RAG_OUTPUT)
    except Exception as e:
        print(f"⚠️ Error saving: {e}")

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

    max_retries = 5
    backoff = 5
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            elif resp.status_code == 429:
                print(f"⚠️ Rate limit hit. Backing off {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
            else:
                print(f"⚠️ HTTP {resp.status_code}: {resp.text[:150]}. Retrying...")
                time.sleep(3)
        except Exception as e:
            print(f"⚠️ Exception: {e}. Retrying...")
            time.sleep(3)

    return "[ERROR: Failed to generate response from Cerebras after multiple retries]"

def main():
    print("🚀 Starting Cerebras AI RAG Answer Generation...")
    print(f"🤖 Model: {MODEL} (120B params — ultra fast!)")

    data = load_data()

    pending_qids = [
        qid for qid, item in data.items()
        if not item.get("rag_answer", "").strip()
        or item.get("rag_answer", "").startswith("[ERROR")
    ]

    total = len(data)
    todo = len(pending_qids)

    print(f"📋 Total Questions: {total}")
    print(f"⏳ Pending Questions: {todo}")
    print(f"⚡ Estimated time at 60 RPM: ~{todo // 60 + 1} minutes")

    if todo == 0:
        print("✅ All questions already answered!")
        return

    t_start = time.time()
    for index, qid in enumerate(pending_qids, 1):
        item = data[qid]
        question = item["question"]

        context_blocks = []
        for j, chunk in enumerate(item.get("retrieved_chunks", [])[:MAX_CHUNKS], 1):
            context_blocks.append(
                f"[{j}] Source: {chunk.get('source_file', '')}, Page: {chunk.get('page', '')}\n"
                f"Text: {chunk.get('text', '')}"
            )
        context_str = "\n\n".join(context_blocks)

        prompt_text = (
            "You are a helpful research assistant. Answer the following question based ONLY on the provided context.\n"
            "CRITICAL RULES:\n"
            "1. Use only the provided context. Do NOT use your own knowledge.\n"
            "2. Output ONLY the direct answer. No introduction, no prefix.\n"
            "3. If the context does not contain enough information, state that clearly.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        t0 = time.time()
        ans = query_cerebras(prompt_text)
        latency = time.time() - t0

        item["rag_answer"] = ans
        item["latency_sec"] = latency
        data[qid] = item
        save_data_atomic(data)

        print(f"⚡ [{index}/{todo}] ✓ {qid} in {latency:.2f}s (len={len(ans)})")

        time.sleep(RATE_LIMIT_DELAY)

    elapsed = time.time() - t_start
    print(f"\n🎉 All done in {elapsed / 60:.2f} minutes!")

if __name__ == "__main__":
    main()
