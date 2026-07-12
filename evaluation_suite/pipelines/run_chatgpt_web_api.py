import os
import sys
import json
import time
import uuid
import requests
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAG_OUTPUT = ROOT / "evaluation_suite" / "rag_retrieved_answers.json"

# ── ChatGPT Internal API Settings ────────────────────────────────────────────
# Paste your Bearer token from Chrome DevTools here or set via env var
BEARER_TOKEN = os.getenv("CHATGPT_BEARER", "")
API_URL = "https://chatgpt.com/backend-api/conversation"
MODEL = "gpt-4o-mini"

# Throttling: 1 request every 2 seconds to avoid detection
RATE_LIMIT_DELAY = 2.0

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

def query_chatgpt(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Origin": "https://chatgpt.com",
        "Referer": "https://chatgpt.com/",
    }

    payload = {
        "action": "next",
        "messages": [
            {
                "id": str(uuid.uuid4()),
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": [prompt]},
            }
        ],
        "model": MODEL,
        "conversation_id": None,
        "parent_message_id": str(uuid.uuid4()),
        "timezone_offset_min": -330,
        "stream": True,
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=30
            )

            if response.status_code == 401:
                print("❌ Bearer token expired or invalid! Please update CHATGPT_BEARER.")
                sys.exit(1)
            elif response.status_code == 429:
                print(f"⚠️ Rate limited. Waiting 30s...")
                time.sleep(30)
                continue
            elif response.status_code != 200:
                print(f"⚠️ HTTP {response.status_code}. Retrying...")
                time.sleep(5)
                continue

            # Parse streamed SSE response
            final_text = ""
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        parts = (
                            data.get("message", {})
                            .get("content", {})
                            .get("parts", [])
                        )
                        if parts and isinstance(parts[0], str):
                            final_text = parts[0]
                    except json.JSONDecodeError:
                        continue

            if final_text.strip():
                return final_text.strip()
            else:
                print(f"⚠️ Empty response on attempt {attempt+1}. Retrying...")
                time.sleep(3)

        except Exception as e:
            print(f"⚠️ Exception: {e}. Retrying...")
            time.sleep(3)

    return "[ERROR: Failed to generate response from ChatGPT after multiple retries]"

def main():
    print("🚀 Starting ChatGPT Web API RAG Answer Generation...")
    print(f"🤖 Model: {MODEL} (via ChatGPT internal API)")

    if not BEARER_TOKEN:
        print("❌ CHATGPT_BEARER environment variable is not set.")
        print("   Get your token from Chrome DevTools > Network > conversation > Headers > Authorization")
        sys.exit(1)

    data = load_data()

    # Identify pending entries
    pending_qids = [
        qid for qid, item in data.items()
        if not item.get("rag_answer", "").strip()
        or item.get("rag_answer", "").startswith("[ERROR")
    ]

    total = len(data)
    todo = len(pending_qids)

    print(f"📋 Total Questions: {total}")
    print(f"⏳ Pending Questions: {todo}")

    if todo == 0:
        print("✅ All questions already answered!")
        return

    t_start = time.time()
    for index, qid in enumerate(pending_qids, 1):
        item = data[qid]
        question = item["question"]

        # Format context chunks
        context_blocks = []
        for j, chunk in enumerate(item.get("retrieved_chunks", []), 1):
            context_blocks.append(
                f"[{j}] Source: {chunk.get('source_file', '')}, Page: {chunk.get('page', '')}\n"
                f"Text: {chunk.get('text', '')}"
            )
        context_str = "\n\n".join(context_blocks)

        prompt_text = (
            "You are a helpful research assistant. Answer the following question based ONLY on the provided context.\n"
            "CRITICAL RULES:\n"
            "1. Use only the provided context. Do NOT use your own knowledge.\n"
            "2. Output ONLY the direct answer. Do NOT include any introduction or prefix.\n"
            "3. If the context does not contain enough information, state that clearly.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        t0 = time.time()
        ans = query_chatgpt(prompt_text)
        latency = time.time() - t0

        # Save progress live and atomically
        item["rag_answer"] = ans
        item["latency_sec"] = latency
        data[qid] = item
        save_data_atomic(data)

        print(f"⚡ [{index}/{todo}] ✓ {qid} in {latency:.2f}s (len={len(ans)})")

        # Throttle
        time.sleep(RATE_LIMIT_DELAY)

    elapsed = time.time() - t_start
    print(f"\n🎉 All done in {elapsed/60:.2f} minutes!")

if __name__ == "__main__":
    main()
