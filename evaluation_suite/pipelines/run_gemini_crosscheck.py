"""Gemini cross-check judge — removes the same-model-judging caveat.

Grades a subset of the stored RAG answers with Gemini 2.5 Flash (via the
OpenRouter API, so no separate Google key is needed) using DeepEval's
FaithfulnessMetric, then reports agreement with the GLM-5.3-Flash judge
scores from judge_results.json.

Usage:
    rag/bin/python3 -m evaluation_suite.pipelines.run_gemini_crosscheck [--n 20]
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "evaluation_suite" / "evaluations" / "llm_judge"
JUDGE_PATH = OUT_DIR / "judge_results.json"
CROSS_PATH = OUT_DIR / "gemini_crosscheck.json"
MODEL = "google/gemini-2.5-flash"


from deepeval.models.base_model import DeepEvalBaseLLM


class GeminiJudge(DeepEvalBaseLLM):
    """DeepEval-compatible judge pinned to Gemini 2.5 Flash via OpenRouter."""

    def __init__(self):
        load_dotenv(ROOT / ".env", override=True)
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise SystemExit("OPENROUTER_API_KEY not set")

    def _call(self, prompt):
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json",
                   "HTTP-Referer": "https://github.com/venkatbayanaboina/research-rag-assistant"}
        t0 = time.time()
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers,
                          json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                                "temperature": 0.0}, timeout=180)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
        return r.json()["choices"][0]["message"]["content"].strip(), time.time() - t0

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        text, _ = self._call(prompt)
        return text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def generate_with_schema(self, prompt, schema):
        text, _ = self._call(prompt + "\n\nReturn ONLY valid JSON matching the requested schema.")
        return text

    async def a_generate_with_schema(self, prompt, schema):
        return self.generate_with_schema(prompt, schema)

    def get_model_name(self):
        return MODEL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="subset size (every k-th case)")
    args = parser.parse_args()

    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import FaithfulnessMetric

    answers = json.loads((OUT_DIR / "rag_answers.json").read_text())
    glm = json.loads(JUDGE_PATH.read_text())
    qids = sorted(answers.keys())[:: max(1, len(answers) // args.n)][: args.n]
    print(f"[crosscheck] {len(qids)} cases with {MODEL}")

    judge = GeminiJudge()
    results = {}
    for i, qid in enumerate(qids):
        entry = answers[qid]
        if entry.get("error") or not entry.get("answer"):
            continue
        tc = LLMTestCase(input=entry["query"], actual_output=entry["answer"],
                         retrieval_context=entry["contexts"] or ["No chunks retrieved."])
        try:
            m = FaithfulnessMetric(threshold=0.6, model=judge, include_reason=True)
            m.measure(tc)
            gem_score = round(float(m.score), 4)
        except Exception as e:
            gem_score = None
            print(f"[crosscheck] {i+1}/{len(qids)} {qid} ERROR {str(e)[:120]}")
            results[qid] = {"gemini_faithfulness": None, "error": str(e)[:160]}
            (OUT_DIR / "gemini_crosscheck.json").write_text(json.dumps(results, indent=2))
            continue
        glm_score = (glm.get(qid) or {}).get("faithfulness")
        results[qid] = {"gemini_faithfulness": gem_score, "glm_faithfulness": glm_score,
                        "reason": (m.reason or "")[:300]}
        (OUT_DIR / "gemini_crosscheck.json").write_text(json.dumps(results, indent=2))
        print(f"[crosscheck] {i+1}/{len(qids)} {qid} gemini={gem_score} glm={glm_score}")
        time.sleep(0.5)

    pairs = [(v["gemini_faithfulness"], v.get("glm_faithfulness"))
             for v in results.values()
             if isinstance(v.get("gemini_faithfulness"), (int, float))
             and isinstance(v.get("glm_faithfulness"), (int, float))]
    if pairs:
        diffs = [abs(g - m) for g, m in pairs]
        agree = sum(1 for g, m in pairs if (g >= 0.6) == (m >= 0.6))
        summary = {
            "model": MODEL, "n_pairs": len(pairs),
            "mean_gemini_faithfulness": round(sum(p[0] for p in pairs) / len(pairs), 4),
            "mean_glm_faithfulness": round(sum(p[1] for p in pairs) / len(pairs), 4),
            "mean_abs_difference": round(sum(diffs) / len(diffs), 4),
            "pass_fail_agreement_at_0.6": f"{agree}/{len(pairs)}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        results["_summary"] = summary
        (OUT_DIR / "gemini_crosscheck.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
