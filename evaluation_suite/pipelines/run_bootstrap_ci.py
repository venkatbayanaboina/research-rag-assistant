"""Bootstrap 95% confidence intervals for the reported evaluation metrics.

Paired bootstrap (B=10,000, fixed seed) over the same 60 gold cases:
  - Hit@1, Recall@5, MRR for dense and hybrid retrieval (per-case ranks from
    the benchmark JSONs — the exact runs behind the reported numbers);
  - the dense-vs-hybrid difference (same resampled case sets in both modes);
  - mean faithfulness over the 60 judged answers.

A difference whose 95% CI excludes 0 is reported as significant; anything
whose CI contains 0 is stated plainly as within noise.

Usage: rag/bin/python3 -m evaluation_suite.pipelines.run_bootstrap_ci
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RB = ROOT / "evaluation_suite" / "evaluations" / "retrieval_benchmark"
LLJ = ROOT / "evaluation_suite" / "evaluations" / "llm_judge"
OUT = ROOT / "evaluation_suite" / "evaluations" / "bootstrap_ci.json"

B = 10_000
SEED = 20260101


def ranks_from(path):
    per_query = json.loads(Path(path).read_text())["per_query"]
    return {q["qid"]: q["first_relevant_rank"] for q in per_query}


def metric_values(ranks):
    """Per-case metric contributions in fixed qid order (for pairing)."""
    qs = sorted(ranks)
    hit1 = [1.0 if ranks[q] == 1 else 0.0 for q in qs]
    rec5 = [1.0 if ranks[q] is not None else 0.0 for q in qs]
    mrr = [1.0 / ranks[q] if ranks[q] is not None else 0.0 for q in qs]
    return qs, {"hit@1": hit1, "recall@5": rec5, "mrr": mrr}


def ci(vals, rng):
    n = len(vals)
    stats = []
    for _ in range(B):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        stats.append(sum(sample) / n)
    stats.sort()
    lo, hi = stats[int(0.025 * B)], stats[int(0.975 * B) - 1]
    return sum(vals) / n, lo, hi


def paired_ci(a_vals, b_vals, rng):
    """CI of mean(b - a) over paired resamples."""
    n = len(a_vals)
    deltas = [b - a for a, b in zip(a_vals, b_vals)]
    stats = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(sum(deltas[i] for i in idx) / n)
    stats.sort()
    lo, hi = stats[int(0.025 * B)], stats[int(0.975 * B) - 1]
    return sum(deltas) / n, lo, hi


def fmt(lo, hi, scale=100):
    return f"[{lo*scale:.1f}%, {hi*scale:.1f}%]" if scale == 100 else f"[{lo:.3f}, {hi:.3f}]"


def main():
    rng = random.Random(SEED)
    dense = ranks_from(RB / "retrieval_benchmark_dense_k5.json")
    hybrid = ranks_from(RB / "retrieval_benchmark_hybrid_k5.json")
    assert set(dense) == set(hybrid), "case sets differ between runs"

    qids, d_metrics = metric_values(dense)
    _, h_metrics = metric_values(hybrid)

    out = {"B": B, "seed": SEED, "n_cases": len(qids), "retrieval": {}, "judge": {}}

    print(f"Paired bootstrap, B={B}, seed={SEED}, n={len(qids)} cases\n")
    for name in ("hit@1", "recall@5", "mrr"):
        dv, dlo, dhi = ci(d_metrics[name], rng)
        hv, hlo, hhi = ci(h_metrics[name], rng)
        dd, dlo2, dhi2 = paired_ci(d_metrics[name], h_metrics[name], rng)
        scale = 100 if name != "mrr" else 1000
        sig = "SIGNIFICANT" if (dlo2 > 0 or dhi2 < 0) else "WITHIN NOISE"
        out["retrieval"][name] = {
            "dense": {"value": round(dv, 4), "ci95": [round(dlo, 4), round(dhi, 4)]},
            "hybrid": {"value": round(hv, 4), "ci95": [round(hlo, 4), round(hhi, 4)]},
            "delta_hybrid_minus_dense": {"mean": round(dd, 4), "ci95": [round(dlo2, 4), round(dhi2, 4)],
                                          "verdict": sig},
        }
        print(f"{name:9s} dense {dv*scale:.1f} {fmt(dlo, dhi)} | "
              f"hybrid {hv*scale:.1f} {fmt(hlo, hhi)} | "
              f"delta {dd*scale:+.1f} {fmt(dlo2, dhi2)} -> {sig}")

    judge = json.loads((LLJ / "judge_results.json").read_text())
    f_vals = [v["faithfulness"] for v in judge.values()
              if isinstance(v.get("faithfulness"), (int, float))]
    r_vals = [v["relevancy"] for v in judge.values()
              if isinstance(v.get("relevancy"), (int, float))]
    for name, vals in (("faithfulness", f_vals), ("relevancy", r_vals)):
        mv, lo, hi = ci(vals, rng)
        out["judge"][name] = {"value": round(mv, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(vals)}
        print(f"{name:9s} mean {mv:.3f} {fmt(lo, hi)} (n={len(vals)})")

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
