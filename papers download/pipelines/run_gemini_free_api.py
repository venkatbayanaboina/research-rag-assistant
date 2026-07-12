import os
import sys
import json
import time
import requests
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAG_OUTPUT = ROOT / "papers download" / "rag_retrieved_answers.json"

# ── Gemini Free API Settings ─────────────────────────────────────────────────
# Retrieve key from environment or use a default if set
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# Throttling to stay under the 15 RPM free limit (1 request every 4.5 seconds = 13.3 RPM)
RATE_LIMIT_DELAY = 4.5

def load_data() -> dict:
    if not RAG_OUTPUT.exists():
        print(f"❌ {RAG_OUTPUT} not found. Make sure run_rag_evaluation.py ran first.")
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
        print(f"⚠️ Error saving data atomically: {e}")
        if temp_path.exists():
            try: os.remove(temp_path)
            except Exception: pass

def query_gemini(prompt: str) -> str:
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
    backoff = 5
    
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
                time.sleep(2)
        except Exception as e:
            print(f"⚠️ Request exception: {e}. Retrying...")
            time.sleep(2)
            
    return "[ERROR: Failed to generate response from Gemini API after multiple retries]"

def main():
    print("🚀 Starting Gemini Free Tier API Generation...")
    print(f"🤖 Model: {MODEL_NAME} (Throttled to stay under 15 RPM)")
    
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
        
    t_start = time.time()
    for index, qid in enumerate(pending_qids, 1):
        item = data[qid]
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
        
        # Save progress live and atomically
        item["rag_answer"] = ans
        item["latency_sec"] = latency
        data[qid] = item
        save_data_atomic(data)
        
        print(f"⚡ [{index}/{todo}] ✓ Answered: {qid} in {latency:.2f}s (Length: {len(ans)})")
        
        # Enforce rate limit (4.5s delay to keep total request rate at 13.3 RPM)
        time.sleep(RATE_LIMIT_DELAY)
        
    elapsed = time.time() - t_start
    print(f"\n🎉 All answers generated successfully in {elapsed/60:.2f} minutes!")

if __name__ == "__main__":
    main()
