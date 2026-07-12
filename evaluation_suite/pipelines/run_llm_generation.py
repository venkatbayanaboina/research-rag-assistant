"""
RAG Pipeline Evaluation — Step 1.5 (LLM Generation)
===================================================
Reads rag_output_queries.json, and for every entry where "rag_answer" is empty,
passes the question and retrieved chunks to the LLM (Gemini) to generate
the answer, then saves the result back in place.

Usage:
    python3 run_llm_generation.py
"""

import os
import sys
import json
import time
from pathlib import Path

# ── Make sure root imports work ──────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.core.generator import generate_answer

# ── Paths ────────────────────────────────────────────────────────────────────
RAG_OUTPUT = ROOT / "evaluation_suite" / "rag_retrieved_answers.json"

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_data() -> dict:
    if not RAG_OUTPUT.exists():
        print(f"❌ {RAG_OUTPUT} not found. Run run_rag_evaluation.py first.")
        sys.exit(1)
    with open(RAG_OUTPUT) as f:
        return json.load(f)

def save_data(data: dict):
    with open(RAG_OUTPUT, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ── Main ─────────────────────────────────────────────────────────────────────

import argparse

def run():
    parser = argparse.ArgumentParser(description="Run RAG Answer Generation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of answers to generate")
    parser.add_argument("--shard", type=int, default=1, help="Shard index (1-indexed)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total shards")
    args = parser.parse_args()

    data = load_data()

    # Find pending generations (where rag_answer is empty or error)
    pending_qids = [
        qid for qid, item in data.items()
        if not item.get("rag_answer", "").strip() or item.get("rag_answer", "").startswith("[ERROR")
    ]

    global RAG_OUTPUT
    if args.num_shards > 1:
        RAG_OUTPUT = RAG_OUTPUT.parent / f"rag_retrieved_answers_shard_{args.shard}.json"
        sharded_qids = []
        for i, qid in enumerate(pending_qids):
            if (i % args.num_shards) == (args.shard - 1):
                sharded_qids.append(qid)
        pending_qids = sharded_qids
        # Only retain the sharded entries to write to the shard-specific file
        data = {qid: data[qid] for qid in pending_qids}
        print(f"分 Sharding Enabled: Running Shard {args.shard}/{args.num_shards}")

    # Apply limit if specified
    if args.limit is not None:
        pending_qids = pending_qids[:args.limit]
        data = {qid: data[qid] for qid in pending_qids}
        print(f"⚠️ Limit enabled: capping generation to the first {args.limit} pending answers.")

    total = len(data)
    todo  = len(pending_qids)
    print(f"Total queries in this shard run : {total}")
    print(f"Remaining answers to generate: {todo}\n")

    if todo == 0:
        print("✅ All RAG answers have already been generated!")
        return

    for i, qid in enumerate(pending_qids, 1):
        item = data[qid]
        question = item["question"]
        print(f"[{i}/{todo}] Generating answer for: {qid}")
        print(f"  Q: {question[:100]}...")

        # Reconstruct standard Search Results dict structure expected by generate_answer
        search_results = []
        for rc in item.get("retrieved_chunks", []):
            search_results.append({
                "score": rc.get("score", 0.0),
                "chunk": {
                    "source_file": rc.get("source_file", ""),
                    "page": rc.get("page", ""),
                    "text": rc.get("text", "")
                }
            })

        t0 = time.time()
        try:
            # Generate LLM answer using retrieved chunks
            rag_answer = generate_answer(question, search_results, image_results=[], chat_history=[])
            print(f"  ✓ Answer generated in {round(time.time() - t0, 2)}s")
        except Exception as e:
            print(f"  ❌ Generation failed: {e}")
            rag_answer = f"[ERROR: {e}]"

        # Update and save
        item["rag_answer"] = rag_answer
        save_data(data)
        time.sleep(12.5)  # Respect 5 RPM free tier limits (12.5s cooldown)

    print(f"\n✅ All LLM answers generated successfully!")
    print(f"   Results updated in: {RAG_OUTPUT}")

if __name__ == "__main__":
    run()
