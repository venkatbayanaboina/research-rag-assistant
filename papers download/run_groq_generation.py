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
RAG_OUTPUT = ROOT / "papers download" / "rag_retrieved_answers.json"

# ── Groq API Settings ────────────────────────────────────────────────────────
# Accepts a comma-separated list of keys, e.g. GROQ_API_KEYS="key1,key2"
API_KEYS_ENV = os.getenv("GROQ_API_KEYS", "") or os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b-versatile"

# Throttling per thread to stay under 12,000 TPM
RATE_LIMIT_DELAY = 7.5

# Thread lock to prevent concurrent write collisions to the main file (using RLock to prevent self-deadlocks)
db_lock = threading.RLock()

def load_data() -> dict:
    if not RAG_OUTPUT.exists():
        print(f"❌ {RAG_OUTPUT} not found. Make sure run_rag_evaluation.py ran first.")
        sys.exit(1)
    with open(RAG_OUTPUT) as f:
        return json.load(f)

def save_data_atomic(data: dict):
    # Atomic write wrapped in a lock to prevent concurrent thread overrides
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

def query_groq(prompt: str, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    
    max_retries = 5
    backoff = 10
    
    for attempt in range(max_retries):
        try:
            response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            elif response.status_code == 429:
                print(f"⚠️ Groq Rate Limit (429) hit. Retrying in {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2
            else:
                print(f"⚠️ Error {response.status_code}: {response.text}. Retrying...")
                time.sleep(5)
        except Exception as e:
            print(f"⚠️ Request exception: {e}. Retrying...")
            time.sleep(5)
            
    return "[ERROR: Failed to generate response from Groq after multiple retries]"

def run_shard(shard_id: int, qids: list, api_key: str, data: dict, total_pending: int):
    print(f"🧵 Thread-{shard_id} launched with {len(qids)} questions using key: ...{api_key[-6:]}")
    
    for index, qid in enumerate(qids, 1):
        item = data[qid]
        question = item["question"]
        
        # Format context
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
        ans = query_groq(prompt_text, api_key)
        latency = time.time() - t0
        
        # Update shared dictionary
        with db_lock:
            item["rag_answer"] = ans
            item["latency_sec"] = latency
            # Read current state of database from disk to keep other thread's progress
            try:
                disk_data = load_data()
                disk_data[qid] = item
                save_data_atomic(disk_data)
            except Exception as e:
                # Fallback to local memory save if disk is temporarily busy
                save_data_atomic(data)
        
        print(f"🧵 Thread-{shard_id} | [{index}/{len(qids)}] ✓ Answered: {qid} in {latency:.2f}s")
        
        # Throttle to stay under TPM
        time.sleep(RATE_LIMIT_DELAY)

def main():
    print("🚀 Starting Groq API Multithreaded RAG Answer Generation...")
    print(f"🤖 Model: {MODEL_NAME}")
    
    if not API_KEYS_ENV:
        print("❌ No GROQ_API_KEYS or GROQ_API_KEY environment variables found.")
        sys.exit(1)
        
    # Split the keys
    api_keys = [k.strip() for k in API_KEYS_ENV.split(",") if k.strip()]
    num_threads = len(api_keys)
    print(f"🔑 Detected {num_threads} API keys. Launching {num_threads} parallel threads...")
    
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
        
    # Shard the pending items across available keys
    shards = [[] for _ in range(num_threads)]
    for i, qid in enumerate(pending_qids):
        shards[i % num_threads].append(qid)
        
    # Run threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for shard_id, (shard_qids, api_key) in enumerate(zip(shards, api_keys), start=1):
            if shard_qids:
                futures.append(
                    executor.submit(run_shard, shard_id, shard_qids, api_key, data, todo)
                )
                
        # Wait for all threads to complete
        for future in futures:
            future.result()
            
    print("\n🎉 All answers generated and saved successfully!")

if __name__ == "__main__":
    main()
