"""
RAG Pipeline Evaluation — Step 1 of 2
=======================================
Runs every question from evaluation_dataset.json through the RAG pipeline,
retrieves chunks from the FAISS vector store, generates an answer via the LLM,
and saves all results to rag_output_queries.json.

Usage:
    python3 run_rag_evaluation.py

Output: rag_output_queries.json
"""

import os
# Disable parallel thread/process forks which cause segmentation faults on macOS
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import json
import time
from pathlib import Path

# ── Make sure root imports work ──────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.core.vector_store import search_store, search_image_store, get_indexed_documents

# ── Paths ────────────────────────────────────────────────────────────────────
EVAL_DATASET  = ROOT / "evaluation_suite" / "gold_qa_dataset.json"
RAG_OUTPUT    = ROOT / "evaluation_suite" / "rag_retrieved_answers.json"

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_existing_output() -> dict:
    if RAG_OUTPUT.exists():
        with open(RAG_OUTPUT) as f:
            return json.load(f)
    return {}

def save_output(data: dict):
    with open(RAG_OUTPUT, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def format_chunks_for_storage(search_results: list) -> list:
    """Serialise retrieved chunks into a compact JSON-safe format."""
    chunks = []
    for r in search_results:
        chunk = r["chunk"]
        chunks.append({
            "score":       round(r["score"], 4),
            "source_file": chunk.get("source_file", ""),
            "page":        chunk.get("page_number", chunk.get("page", "")),
            "text":        chunk.get("text", "")[:800],   # keep first 800 chars for readability
        })
    return chunks

# ── Main ─────────────────────────────────────────────────────────────────────

import argparse

def run():
    parser = argparse.ArgumentParser(description="Run RAG Evaluation Retrieval")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of QA pairs to retrieve")
    args = parser.parse_args()

    if not EVAL_DATASET.exists():
        print(f"❌ {EVAL_DATASET} not found. Run the QA generation step first.")
        sys.exit(1)

    with open(EVAL_DATASET) as f:
        dataset: dict = json.load(f)

    existing = load_existing_output()

    indexed_docs = get_indexed_documents()
    print(f"Indexed documents in FAISS: {len(indexed_docs)}")

    # Flatten all QA pairs
    all_pairs = []
    for paper_id, paper_data in dataset.items():
        for qa in paper_data.get("qa_pairs", []):
            all_pairs.append((paper_data["paper_title"], qa))

    # Apply limit if specified
    if args.limit is not None:
        all_pairs = all_pairs[:args.limit]
        print(f"⚠️ Limit enabled: evaluation capped to the first {args.limit} QA pairs.")

    total = len(all_pairs)
    done  = sum(1 for _, qa in all_pairs if qa["question_id"] in existing)
    print(f"Total QA pairs : {total}")
    print(f"Already done   : {done}")
    print(f"Remaining      : {total - done}\n")

    for i, (paper_title, qa) in enumerate(all_pairs, 1):
        qid = qa["question_id"]

        if qid in existing:
            continue

        question = qa["question"]
        print(f"[{i}/{total}] {qid}")
        print(f"  Q: {question[:100]}...")

        t0 = time.time()

        # ── Step 1: Retrieve chunks ───────────────────────────────────────
        try:
            search_results = search_store(question, rerank=config.RERANK_ENABLED)
        except Exception as e:
            print(f"  ⚠️ Retrieval failed: {e}")
            search_results = []

        try:
            image_results = search_image_store(question)
        except Exception as e:
            image_results = []

        elapsed = round(time.time() - t0, 2)
        print(f"  ✓ Done in {elapsed}s  |  {len(search_results)} chunks retrieved")

        # ── Step 3: Store result (rag_answer is populated later) ──────────
        existing[qid] = {
            "question_id":      qid,
            "paper_title":      paper_title,
            "question_type":    qa.get("question_type", ""),
            "difficulty":       qa.get("difficulty", ""),
            "question":         question,
            "expected_answer":  qa["expected_answer"],
            "evidence":         qa.get("evidence", {}),
            "rag_answer":       "",
            "retrieved_chunks": format_chunks_for_storage(search_results),
            "latency_sec":      elapsed,
        }

        save_output(existing)

    print(f"\n✅ RAG output complete! Results saved to: {RAG_OUTPUT}")
    print(f"   Total processed: {len(existing)}/{total}")

if __name__ == "__main__":
    run()
