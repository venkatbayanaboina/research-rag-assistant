"""Reproducible retrieval benchmark over the gold QA dataset.

Compares text retrieval configurations on the same 60 QA cases (4 indexed
papers from evaluation_suite/dataset/gold_qa_dataset.json). Relevance label =
a retrieved chunk whose source_file matches the case paper and whose page
equals the QA evidence page. No LLM is involved: results are fully
reproducible offline.

Usage (from repo root, with the rag venv):
    rag/bin/python3 -m evaluation_suite.pipelines.run_retrieval_benchmark \
        --mode dense   --k 5
    rag/bin/python3 -m evaluation_suite.pipelines.run_retrieval_benchmark \
        --mode hybrid  --k 5

Outputs JSON + Markdown to evaluation_suite/evaluations/retrieval_benchmark/.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: F401  (imports establish repo-root-relative storage paths)
from src.core.vector_store import search_store

GOLD_PATH = ROOT / "evaluation_suite" / "dataset" / "gold_qa_dataset.json"
OUT_DIR = ROOT / "evaluation_suite" / "evaluations" / "retrieval_benchmark"


def load_cases():
    gold = json.loads(GOLD_PATH.read_text())
    registry = json.loads((ROOT / "storage" / "chunks.json").read_text())
    docs = {c["source_file"] for c in registry}
    cases = []
    for arxiv_id, entry in gold.items():
        filename = next((f for f in docs if f.startswith(arxiv_id)), None)
        if not filename:
            continue
        for qa in entry["qa_pairs"]:
            page = (qa.get("evidence") or {}).get("page")
            if page:
                cases.append({
                    "qid": qa["question_id"],
                    "type": qa.get("question_type", "?"),
                    "query": qa["question"],
                    "doc": filename,
                    "evidence_page": page,
                })
    return cases


def run(mode, k):
    cases = load_cases()
    print(f"[benchmark] mode={mode} k={k} cases={len(cases)}")

    hit1 = recall = 0.0
    recall3 = 0.0
    mrr = 0.0
    per_type = {}
    latencies = []
    per_query = []
    misses = []

    for i, case in enumerate(cases):
        t0 = time.time()
        results = search_store(case["query"], k=k, rerank=False, mode=mode)
        elapsed_ms = (time.time() - t0) * 1000
        latencies.append(elapsed_ms)

        same_doc = [r for r in results if r["chunk"]["source_file"] == case["doc"]]
        ranks = [j + 1 for j, r in enumerate(same_doc)
                 if r["chunk"]["page"] == case["evidence_page"]]

        h1 = 1 if ranks and ranks[0] == 1 else 0
        rec = 1 if ranks else 0
        rec3 = 1 if any(r <= 3 for r in ranks) else 0
        rr = 1.0 / ranks[0] if ranks else 0.0
        hit1 += h1
        recall += rec
        recall3 += rec3
        mrr += rr

        stats = per_type.setdefault(case["type"], {"n": 0, "hit1": 0, "recall": 0, "rr": 0.0})
        stats["n"] += 1
        stats["hit1"] += h1
        stats["recall"] += rec
        stats["rr"] += rr

        per_query.append({"qid": case["qid"], "type": case["type"],
                          "latency_ms": round(elapsed_ms, 1),
                          "first_relevant_rank": ranks[0] if ranks else None})

        if not ranks:
            misses.append({"qid": case["qid"], "type": case["type"], "query": case["query"][:120]})

        if (i + 1) % 10 == 0:
            print(f"[benchmark] {i + 1}/{len(cases)} processed")

    n = len(cases)
    lat_sorted = sorted(latencies)
    summary = {
        "mode": mode,
        "k": k,
        "rerank": False,
        "n_cases": n,
        "hit_at_1": round(hit1 / n, 4),
        f"recall_at_3": round(recall3 / n, 4),
        f"recall_at_{k}": round(recall / n, 4),
        "mrr": round(mrr / n, 4),
        "latency_ms_mean": round(sum(latencies) / len(latencies), 1),
        "latency_ms_p50": round(lat_sorted[len(lat_sorted) // 2], 1),
        "latency_ms_p95": round(lat_sorted[int(len(lat_sorted) * 0.95) - 1], 1),
        "per_type": {
            t: {"n": s["n"],
                "hit_at_1": round(s["hit1"] / s["n"], 4),
                f"recall_at_{k}": round(s["recall"] / s["n"], 4),
                "mrr": round(s["rr"] / s["n"], 4)}
            for t, s in sorted(per_type.items())
        },
        "per_query": per_query,
        "missed_qids": [m["qid"] for m in misses],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return summary, misses


def save(summary, misses):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"retrieval_benchmark_{summary['mode']}_k{summary['k']}.json"
    json_path.write_text(json.dumps(summary, indent=2))

    k = summary["k"]
    lines = [
        f"# Retrieval Benchmark — mode={summary['mode']}, k={k}, rerank=off",
        "",
        f"- Cases: {summary['n_cases']} (gold QA pairs for 4 indexed papers)",
        f"- Hit@1: **{summary['hit_at_1']*100:.2f}%**",
        f"- Recall@3: **{summary['recall_at_3']*100:.2f}%**",
        f"- Recall@{k}: **{summary[f'recall_at_{k}']*100:.2f}%**",
        f"- MRR: **{summary['mrr']:.4f}**",
        f"- Latency/query: mean {summary['latency_ms_mean']} ms · "
        f"p50 {summary['latency_ms_p50']} ms · p95 {summary['latency_ms_p95']} ms "
        "(embedding + search, model warm)",
        "",
        "| Type | n | Hit@1 | Recall@" + str(k) + " | MRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for t, s in summary["per_type"].items():
        lines.append(f"| {t} | {s['n']} | {s['hit_at_1']*100:.2f}% | "
                     f"{s[f'recall_at_{k}']*100:.2f}% | {s['mrr']:.4f} |")
    lines += ["", "## Missed cases (no evidence-page chunk in top-%d)" % k, ""]
    lines += [f"- `{m['qid']}` ({m['type']}): {m['query']}" for m in misses]
    (OUT_DIR / f"retrieval_benchmark_{summary['mode']}_k{summary['k']}.md").write_text("\n".join(lines))
    print(f"[benchmark] saved -> {OUT_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dense", "hybrid"], required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    summary, misses = run(args.mode, args.k)
    print(json.dumps({key: val for key, val in summary.items() if key != "missed_qids"}, indent=2))
    save(summary, misses)


if __name__ == "__main__":
    main()
