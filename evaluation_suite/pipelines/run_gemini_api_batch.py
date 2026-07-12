import os
import sys
import json
import time
import requests
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAG_OUTPUT = ROOT / "evaluation_suite" / "rag_retrieved_answers.json"

# ── Gemini API Settings ──────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# Throttling settings for Paid Tier (1,000 RPM limit -> we target 500 RPM for safety)
MAX_WORKERS = 30
REQUEST_INTERVAL = 0.12  # (60s / 500 requests) = 0.12s minimum delay between launches

# Thread lock to prevent concurrent write collisions to the main file
db_lock = threading.Lock()
rate_limit_lock = threading.Lock()
last_request_time = 0.0

def load_data() -> dict:
    if not RAG_OUTPUT.exists():
        print(f"❌ {RAG_OUTPUT} not found. Make sure run_rag_evaluation.py ran first.")
        sys.exit(1)
    with open(RAG_OUTPUT) as f:
        return json.load(f)

def save_data_atomic(data: dict):
    temp_path = RAG_OUTPUT.with_suffix(".tmp")
    with db_lock:
        try:
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, RAG_OUTPUT)
        except Exception as e:
            print(f"⚠️ Error saving data atomically: {e}")
            if temp_path.exists():
                try: os.remove(temp_path)
                except Exception: pass

def enforce_rate_limit():
    global last_request_time
    with rate_limit_lock:
        now = time.time()
        elapsed = now - last_request_time
        if elapsed < REQUEST_INTERVAL:
            sleep_time = REQUEST_INTERVAL - elapsed
            time.sleep(sleep_time)
        last_request_time = time.time()

def query_gemini(prompt: str) -> str:
    enforce_rate_limit()
    
    params = {"key": GEMINI_API_KEY}
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.3
        }
    }
    
    max_retries = 5
    backoff = 2
    
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, params=params, headers=headers, json=payload, timeout=20)
            if response.status_code == 200:
                resp_json = response.json()
                return resp_json["candidates"][0]["content"]["parts"][0]["text"].strip()
            elif response.status_code == 429:
                print(f"⚠️ Rate Limit (429) hit. Backing off for {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
            else:
                print(f"⚠️ HTTP {response.status_code}: {response.text}. Retrying...")
                time.sleep(1)
        except Exception as e:
            print(f"⚠️ Request exception: {e}. Retrying...")
            time.sleep(1)
            
    return "[ERROR: Failed to generate response from Gemini API after multiple retries]"

def process_query(qid: str, item: dict, index: int, total: int, shared_data: dict):
    question = item["question"]
    
    # Format context chunks
    context_blocks = []
    for j, chunk in enumerate(item.get("retrieved_chunks", []), 1):
        context_blocks.append(
            f"[{j}] Source: {chunk.get('source_file','')}, Page: {chunk.get('page','')}\n"
            f"Text: {chunk.get('text','')}"
        )
    context_str = "\n\n".join(context_blocks)
    
    prompt_text = (
        "You are a helpful research assistant. Answer the following question based ONLY on the provided context.\n"
        "CRITICAL RULES:\n"
        "1. Do NOT search the web. Use only the provided context.\n"
        "2. Output ONLY the direct answer. Do NOT include any conversational introduction, summary, or prefix "
        "(e.g., do NOT say 'Based on the context' or 'Here is your answer'). Start directly with the factual response.\n"
        "3. If the context does not contain enough information to answer, state that clearly.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    
    t0 = time.time()
    ans = query_gemini(prompt_text)
    latency = time.time() - t0
    
    # Update shared data and write to disk
    with db_lock:
        item["rag_answer"] = ans
        item["latency_sec"] = latency
        try:
            disk_data = load_data()
            disk_data[qid] = item
            save_data_atomic(disk_data)
        except Exception:
            save_data_atomic(shared_data)
            
    print(f"⚡ [{index}/{total}] ✓ Answered: {qid} in {latency:.2f}s (Length: {len(ans)})")

def main():
    print("🚀 Starting High-Speed Gemini API Batch Generation (Paid Tier)...")
    print(f"🤖 Model: {MODEL_NAME} (Max Workers: {MAX_WORKERS}, Target: ~500 RPM)")
    
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)
        
    data = load_data()
    
    # Identify pending entries
    pending_qids = [
        qid for qid, item in data.items()
        if not item.get("rag_answer", "").strip() or item.get("rag_answer", "").startswith("[ERROR")
    ]
    
    total = len(data)
    todo = len(pending_qids)
    
    print(f"📋 Total Questions: {total}")
    print(f"⏳ Pending Questions: {todo}")
    
    if todo == 0:
        print("✅ All questions have already been answered!")
        return
        
    # Start high-speed parallel run
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for index, qid in enumerate(pending_qids, start=1):
            futures.append(
                executor.submit(process_query, qid, data[qid], index, todo, data)
            )
            
        # Wait for all queries to finish
        for future in futures:
            future.result()
            
    elapsed = time.time() - t_start
    print(f"\n🎉 All answers generated successfully in {elapsed/60:.2f} minutes!")

if __name__ == "__main__":
    main()
