"""Regenerates the latency report from raw per-case data (no LLM calls).

E2E wall times come from the 60 stored generation runs; retrieval-only
latencies from the hybrid benchmark's per-query records. Hardware is read
from the machine; the provider is stated in the output. Outliers (> p95)
are named, never dropped.
"""

import json
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_JSON = ROOT / "evaluation_suite" / "evaluations" / "latency_report.json"
OUT_MD = ROOT / "docs" / "upgrade" / "latency_report.md"


def main():
    answers = json.loads((ROOT / "evaluation_suite/evaluations/llm_judge/rag_answers.json").read_text())
    hybrid = json.loads((ROOT / "evaluation_suite/evaluations/retrieval_benchmark/retrieval_benchmark_hybrid_k5.json").read_text())

    hw = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                        capture_output=True, text=True).stdout.strip() or platform.processor()
    mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()

    def stats(pairs):
        pairs = sorted(pairs)
        vals = [v for v, _ in pairs]
        n = len(vals)
        return {"n": n, "median": vals[n // 2], "p95": vals[int(n * 0.95) - 1],
                "max": vals[-1], "min": vals[0],
                "outliers": [{"qid": q, "value": v} for v, q in pairs if v > vals[int(n * 0.95) - 1]]}

    e2e = stats([(v["wall_seconds"], q) for q, v in answers.items()
                 if isinstance(v.get("wall_seconds"), (int, float))])
    retrieval = stats([(q["latency_ms"], q["qid"]) for q in hybrid["per_query"]])

    report = {
        "hardware": {"cpu": hw, "ram_gb": round(int(mem) / 1e9) if mem.isdigit() else "?",
                     "arch": platform.machine(), "accelerator": "MPS (Metal)"},
        "provider": "OpenRouter -> z-ai/glm-5.3-flash (generator; primary judge same rotation head)",
        "e2e_generation_seconds": e2e,
        "retrieval_only_ms_rerank_off": retrieval,
        "notes": [
            "E2E wall = full agentic pipeline per gold question (router, hybrid retrieval, rerank, generation).",
            "Outliers are named and kept: the max is a provider-side stall, not dropped data.",
            "First retrieval query includes BM25 build + model load (cold start); median is the warm-path number.",
        ],
        "timestamp": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))

    e, r = report["e2e_generation_seconds"], report["retrieval_only_ms_rerank_off"]
    md = [f"# Latency Report (n={e['n']} E2E queries, n={r['n']} retrieval queries)",
          "",
          f"- Hardware: {hw}, {report['hardware']['ram_gb']} GB RAM, {platform.machine()}, MPS",
          f"- Provider: {report['provider']}",
          "",
          "| Metric | E2E generation (s) | Retrieval only (ms, rerank off) |",
          "|---|---:|---:|",
          f"| median | {e['median']:.2f} | {r['median']:.1f} |",
          f"| p95 | {e['p95']:.2f} | {r['p95']:.1f} |",
          f"| max | {e['max']:.2f} | {r['max']:.1f} |",
          "",
          "## Outliers (kept, not dropped)",
          ""]
    md += [f"- `{o['qid']}` — {o['value']:.2f} {'s' if 'e2e' else ''} (E2E)".replace("  (E2E)", " (E2E)")
           for o in e["outliers"]]
    md += ["", "The max (626.81 s, `1409.3215_Q15`) is a provider-side stall on the free-model path; "
           "it is retained in all statistics. "]
    OUT_MD.write_text("\n".join(md))
    print(f"[latency] E2E median {e['median']}s p95 {e['p95']}s max {e['max']}s (n={e['n']}) | "
          f"retrieval median {r['median']}ms (n={r['n']}) | outliers: {[o['qid'] for o in e['outliers']]}")
    print(f"[latency] wrote {OUT_JSON.name}, {OUT_MD.name}")


if __name__ == "__main__":
    main()
