import os
import json
import sys
from collections import defaultdict

# Include root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

LOG_FILE = os.path.join(config.STORAGE_DIR, "performance_logs.jsonl")

def analyze_latency():
    """
    Reads performance_logs.jsonl and outputs a structured latency analysis report.
    """
    if not os.path.exists(LOG_FILE):
        print("\n❌ No performance logs found. Please ingest a PDF or ask a query to generate log entries first.")
        return

    step_data = defaultdict(list)
    
    with open(LOG_FILE, "r") as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    step = entry.get("step")
                    duration = entry.get("duration_ms")
                    if step and duration is not None:
                        step_data[step].append(duration)
                except Exception:
                    continue

    if not step_data:
        print("\n❌ Performance logs are empty or corrupted.")
        return

    print("\n" + "="*80)
    print("📊 RAG PIPELINE LATENCY PERFORMANCE REPORT")
    print("="*80)
    
    # Print formatted table header
    print(f"{'Pipeline Step':<32} | {'Runs':<6} | {'Avg Latency':<12} | {'Min Latency':<12} | {'Max Latency':<12}")
    print("-"*80)

    # Sort steps logically (Ingestion steps first, then Retrieval, then Generation)
    logical_order = [
        "layout_selection_and_parsing",
        "text_chunking",
        "visual_chunking",
        "text_embedding_generation",
        "visual_embedding_generation",
        "text_retrieval_faiss",
        "visual_retrieval_clip",
        "reranking",
        "llm_answer_generation",
        "fallback_answer_generation"
    ]
    
    # Add any extra steps not in the logical list
    for step in sorted(step_data.keys()):
        if step not in logical_order:
            logical_order.append(step)

    for step in logical_order:
        durations = step_data.get(step)
        if not durations:
            continue
            
        runs = len(durations)
        avg_lat = sum(durations) / runs
        min_lat = min(durations)
        max_lat = max(durations)
        
        print(f"{step:<32} | {runs:<6} | {avg_lat:>9.2f}ms | {min_lat:>9.2f}ms | {max_lat:>9.2f}ms")

    print("="*80)
    print(f"📁 Raw logs stored at: {LOG_FILE}\n")

if __name__ == "__main__":
    analyze_latency()
