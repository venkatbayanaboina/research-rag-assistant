"""End-to-end pipeline latency benchmark (live LLM, fixed query set).

Measures wall-clock time for full agentic pipeline runs — routing, query
expansion, hybrid retrieval, reranking, generation — with the OpenRouter gate
forced so the provider is identical across runs. Model warm-up (BGE /
cross-encoder / CLIP loading) happens on a discarded warm-up query and is
excluded from timings.

Usage:
    rag/bin/python3 -m evaluation_suite.pipelines.run_pipeline_latency \
        --label run1_pre     # before optimizations
    rag/bin/python3 -m evaluation_suite.pipelines.run_pipeline_latency \
        --label run2_post    # after optimizations
"""

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

QUERIES = [
    "What is the primary purpose of convolutional neural networks in the sentence classification paper?",
    "How does target propagation provide credit assignment in deep networks?",
    "What is multi-head attention in the transformer architecture?",
    "What evaluation metrics are reported in the sequence to sequence learning paper?",
]

OUT_DIR = ROOT / "evaluation_suite" / "evaluations" / "performance_benchmark"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="run1_pre | run2_post | ...")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(ROOT))

    from src.core.generator import execute_rag_pipeline
    from src.core.vector_store import get_indexed_documents, search_store, search_image_store
    from src.core.generator.gates.controller import switch_active_gate_to_openrouter

    indexed = get_indexed_documents()
    print(f"[latency] label={args.label} indexed_docs={len(indexed)}")
    if not indexed:
        raise SystemExit("No documents indexed in the global store.")

    # Provider pinning: identical path before/after so the comparison is fair.
    switch_active_gate_to_openrouter()

    # Warm-up: load BGE, CLIP, cross-encoder, BM25 (discarded from timing).
    t0 = time.time()
    search_store("warmup query for model loading", k=5, rerank=True)
    search_image_store("warmup query for model loading")
    print(f"[latency] models warmed in {time.time()-t0:.1f}s (excluded)")

    runs = []
    for i, query in enumerate(QUERIES):
        t0 = time.time()
        result = execute_rag_pipeline(query, indexed)
        wall = time.time() - t0
        runs.append({
            "query": query,
            "wall_seconds": round(wall, 2),
            "model_call_count": result.get("model_call_count"),
            "refinement_iterations": result.get("refinement_iterations"),
            "evidence_status": result.get("evidence_status"),
            "verification_status": (result.get("verification") or {}).get("status"),
            "answer_chars": len(result.get("answer") or ""),
        })
        print(f"[latency] {i+1}/{len(QUERIES)} {wall:6.2f}s  "
              f"llm_calls={result.get('model_call_count')} "
              f"evidence={result.get('evidence_status')} "
              f"verify={(result.get('verification') or {}).get('status')}")

    walls = sorted(r["wall_seconds"] for r in runs)
    summary = {
        "label": args.label,
        "provider": "openrouter (forced, free models)",
        "n_queries": len(runs),
        "wall_seconds_mean": round(sum(walls) / len(walls), 2),
        "wall_seconds_min": walls[0],
        "wall_seconds_max": walls[-1],
        "model_calls_total": sum(r["model_call_count"] or 0 for r in runs),
        "runs": runs,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"pipeline_latency_{args.label}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[latency] mean {summary['wall_seconds_mean']}s  "
          f"total llm calls {summary['model_calls_total']}  saved -> {out}")


if __name__ == "__main__":
    main()
