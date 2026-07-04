import os
import sys
import json
import time
import requests
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAG_OUTPUT = ROOT / "papers download" / "rag_retrieved_answers.json"

# ── Groq API Settings ────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY environment variable is not set. Please set it before running.")
    sys.exit(1)

# Throttling to stay under 12,000 TPM (Token limit per minute)
# ~1.2k tokens per query -> 8 queries per minute (7.5 seconds delay between queries)
RATE_LIMIT_DELAY = 7.5

def load_data() -> dict:
    if not RAG_OUTPUT.exists():
        print(f"❌ {RAG_OUTPUT} not found. Make sure run_rag_evaluation.py ran first.")
        sys.exit(1)
    with open(RAG_OUTPUT) as f:
        return json.load(f)

def save_data(data: dict):
    # Save directly to main file
    with open(RAG_OUTPUT, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def query_groq(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    
    # Retry loop for rate limits (429) or transient network errors
    max_retries = 5
    backoff = 10
    
    for attempt in range(max_retries):
        try:
            response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            elif response.status_code == 429:
                # Rate limit hit
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

def main():
    print("🚀 Starting Groq API RAG Answer Generation...")
    print(f"🤖 Model: {MODEL_NAME}")
    
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
        
    for i, qid in enumerate(pending_qids, 1):
        item = data[qid]
        question = item["question"]
        
        # Format RAG context block
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
        
        print(f"\n[{i}/{todo}] Generating for: {qid}")
        print(f"  Q: {question[:100]}...")
        
        t0 = time.time()
        ans = query_groq(prompt_text)
        latency = time.time() - t0
        
        # Save progress live
        item["rag_answer"] = ans
        item["latency_sec"] = latency
        save_data(data)
        
        print(f"  ✓ Answer generated in {latency:.2f}s (Length: {len(ans)})")
        
        # Throttle to respect TPM limits
        time.sleep(RATE_LIMIT_DELAY)
        
    print("\n🎉 All answers generated and saved successfully!")

if __name__ == "__main__":
    main()
