"""Live latency micro-benchmark for candidate OpenRouter free models.

Sends a small, fixed RAG-style prompt to each candidate model and records
wall-clock time per call. Results drive the gate's model-rotation order
(fastest validated model first) and are stored for the performance report.

Usage:
    rag/bin/python3 -m evaluation_suite.pipelines.benchmark_openrouter_models --label <name>
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "evaluation_suite" / "evaluations" / "performance_benchmark"

PROMPT = ("You are a research assistant. Answer in exactly one short sentence: "
          "What is self-attention in the Transformer architecture?")

CANDIDATES = [
    "z-ai/glm-5.3-flash",                     # cheap ($0.15/$0.50 per M), multimodal, JSON mode
    "nvidia/nemotron-3.5-lightning:free",     # current free rotation head
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-31b-it:free",
]

CONFIRM_EXTRA_CALLS = 1  # additional call for the two fastest models


def call_model(model, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/venkatbayanaboina/research-rag-assistant",
        "X-Title": "Multi-PDF RAG Assistant",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.0,
    }
    t0 = time.time()
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                          headers=headers, json=payload, timeout=120)
        elapsed = time.time() - t0
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"].strip()
            return {"ok": True, "seconds": round(elapsed, 2),
                    "chars": len(text), "preview": text[:80]}
        return {"ok": False, "seconds": round(elapsed, 2),
                "error": f"HTTP {r.status_code}: {r.text[:140]}"}
    except Exception as e:
        return {"ok": False, "seconds": round(time.time() - t0, 2), "error": str(e)[:140]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="models")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=True)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set.")

    results = []
    for model in CANDIDATES:
        calls = [call_model(model, api_key)]
        print(f"{model:45s} call1: {calls[0]['seconds']:6.2f}s "
              f"{'ok ' + str(calls[0].get('chars', 0)) + ' chars' if calls[0]['ok'] else calls[0].get('error', '')[:70]}")
        results.append({"model": model, "calls": calls})
        time.sleep(1.0)

    # Confirm the two fastest valid models with one extra call each.
    valid = [r for r in results if r["calls"][0]["ok"]]
    valid.sort(key=lambda r: r["calls"][0]["seconds"])
    for r in valid[:2]:
        extra = call_model(r["model"], api_key)
        r["calls"].append(extra)
        print(f"{'confirm ' + r['model']:45s} call2: {extra['seconds']:6.2f}s "
              f"{'ok' if extra['ok'] else extra.get('error', '')[:70]}")
        time.sleep(1.0)

    summary = {
        "label": args.label,
        "prompt_chars": len(PROMPT),
        "results": results,
        "ranking_fastest_first": [
            {"model": r["model"],
             "best_seconds": min(c["seconds"] for c in r["calls"] if c["ok"]),
             "calls": len(r["calls"])}
            for r in valid[:3]
        ],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"model_benchmark_{args.label}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {out}")
    print("Ranking (fastest valid first):")
    for r in summary["ranking_fastest_first"]:
        print(f"  {r['best_seconds']:6.2f}s  {r['model']}")


if __name__ == "__main__":
    main()
