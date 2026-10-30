"""Full LLM-judge evaluation of the RAG pipeline (Phase 7).

Two stages, checkpointed so any chunk can be re-run after a timeout:

  stage=generate  — runs the REAL agentic pipeline per gold QA case and stores
                    the answer, retrieval context, and the deterministic
                    citation-verification report (free metrics).
  stage=judge     — grades each stored answer with DeepEval: FaithfulnessMetric
                    (claim-level hallucination check) and a GEval-based Answer
                    Relevancy metric, both judged by the gate chain's current
                    head model.

Cost notes: GLM-5.3-Flash as generator+judge ≈ $0.30–0.50 for the full 60.
Usage metadata is not exposed by the gates, so costs are estimated from
character counts (≈4 chars/token) and clearly labelled as estimates.

Usage:
  rag/bin/python3 -m evaluation_suite.pipelines.run_full_evaluation \
      stage=generate start=0 end=10
  rag/bin/python3 -m evaluation_suite.pipelines.run_full_evaluation stage=judge start=0 end=10
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "evaluation_suite" / "evaluations" / "llm_judge"
ANSWERS_PATH = OUT_DIR / "rag_answers.json"
JUDGE_PATH = OUT_DIR / "judge_results.json"


def load_cases():
    sys.path.insert(0, str(ROOT))
    from evaluation_suite.pipelines.run_retrieval_benchmark import load_cases
    return load_cases()


# ------------------------------------------------------------- stage: generate

def stage_generate(start, end):
    from src.core.generator import execute_rag_pipeline
    from src.core.vector_store import get_indexed_documents

    cases = load_cases()[start:end]
    indexed = get_indexed_documents()
    store = json.loads(ANSWERS_PATH.read_text()) if ANSWERS_PATH.exists() else {}
    print(f"[generate] cases {start}..{end} ({len(cases)}), {len(store)} already stored")

    for i, case in enumerate(cases):
        if case["qid"] in store:
            continue
        t0 = time.time()
        try:
            result = execute_rag_pipeline(case["query"], indexed)
            contexts = [r["chunk"]["text"] for r in result.get("search_results", [])]
            entry = {
                "qid": case["qid"], "type": case["type"], "query": case["query"],
                "answer": result.get("answer", ""),
                "contexts": contexts,
                "evidence_status": result.get("evidence_status"),
                "verification": result.get("verification"),
                "intent": result.get("intent"),
                "model_call_count": result.get("model_call_count"),
                "wall_seconds": round(time.time() - t0, 2),
                "answer_chars": len(result.get("answer") or ""),
                "context_chars": sum(len(c) for c in contexts),
            }
        except Exception as e:
            entry = {"qid": case["qid"], "type": case["type"], "query": case["query"],
                     "answer": "", "contexts": [], "error": str(e)[:200]}
        store[case["qid"]] = entry
        ANSWERS_PATH.write_text(json.dumps(store, indent=2))
        est_tokens = (entry.get("answer_chars", 0) + entry.get("context_chars", 0)) // 4
        print(f"[generate] {i+1}/{len(cases)} {case['qid']} wall={entry.get('wall_seconds')}s "
              f"verify={entry.get('verification', {}).get('status') if entry.get('verification') else 'ERR'} "
              f"~{est_tokens} tok")

    done = sum(1 for c in cases if c["qid"] in store)
    print(f"[generate] {done}/{len(cases)} stored -> {ANSWERS_PATH.name}")


# --------------------------------------------------------------- stage: judge

from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv


class OpenRouterPinnedJudge(DeepEvalBaseLLM):
    """Judge pinned to one OpenRouter model slug (independent-family option,
    config.JUDGE_MODEL). Supports any multimodal/text slug; JSON mode for
    schema calls."""

    def __init__(self, model_slug):
        import requests as _requests
        self.requests = _requests
        self.model_slug = model_slug
        load_dotenv(ROOT / ".env", override=True)
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise SystemExit("OPENROUTER_API_KEY not set")

    def _call(self, prompt):
        r = self.requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model_slug,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.0},
            timeout=180)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
        return r.json()["choices"][0]["message"]["content"].strip()

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        return self._call(prompt)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def generate_with_schema(self, prompt, schema):
        return self._call(prompt + "\n\nReturn ONLY valid JSON matching the requested schema.")

    async def a_generate_with_schema(self, prompt, schema):
        return self.generate_with_schema(prompt, schema)

    def get_model_name(self):
        return self.model_slug


def _relevancy_judge(question, answer, pinned_judge=None):
    """Direct LLM-judge for answer relevancy. GEval is incompatible with our
    custom gate judge in deepeval 4.0.7 ('str' has no attribute 'value'), and
    the native AnswerRelevancyMetric requires an OpenAI embedder — so this
    metric is a single structured gate call with the same judge model."""
    from src.core.generator.client import generate_content_with_retry
    prompt = f"""Rate how well the ANSWER addresses the QUESTION on a scale of 0.0 to 1.0.
1.0 = directly and completely answers it; 0.5 = partially or evasively; 0.0 = unrelated.

QUESTION: {question}

ANSWER: {answer}

Return ONLY JSON: {{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}"""
    if pinned_judge is not None:
        clean = pinned_judge.generate(prompt).strip()
    else:
        clean = generate_content_with_retry(prompt, temperature=0.0,
                                            response_mime_type="application/json").strip()
    if "```" in clean:
        clean = clean.replace("```json", "").replace("```", "").strip()
    data = json.loads(clean)
    score = max(0.0, min(1.0, float(data.get("score", 0.0))))
    return score, str(data.get("reason", ""))[:400]


def stage_judge(start, end, judge_model=None):
    """judge_model=None  — gate rotation head, results -> judge_results.json.
    judge_model=slug — pinned independent model (config.JUDGE_MODEL /
    JUDGE_MODEL env), results -> judge_results_independent.json, reported
    separately from the same-model judge."""
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import FaithfulnessMetric
    from src.core.generator.client import generate_content_with_retry

    independent = bool(judge_model)
    if independent:
        judge = OpenRouterPinnedJudge(judge_model)
        results_path = ROOT / "evaluation_suite/evaluations/llm_judge/judge_results_independent.json"
    else:
        from tests.test_rag_eval import LLMGateJudge
        judge = LLMGateJudge()
        results_path = JUDGE_PATH
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    cases = load_cases()[start:end]
    print(f"[judge] cases {start}..{end} ({len(cases)}), judge model: {judge.get_model_name()}")

    for i, case in enumerate(cases):
        qid = case["qid"]
        if qid in results:
            continue
        # Fresh read per case: generation may still be writing answers in a
        # parallel run. Missing answers are left unjudged (re-run later) rather
        # than permanently marked skipped.
        store = json.loads(ANSWERS_PATH.read_text()) if ANSWERS_PATH.exists() else {}
        entry = store.get(qid)
        if not entry:
            print(f"[judge] {i+1}/{len(cases)} {qid} — answer not generated yet, deferring")
            continue
        if entry.get("error") or not entry.get("answer"):
            results[qid] = {"faithfulness": None, "relevancy": None, "skipped": "no answer"}
            results_path.write_text(json.dumps(results, indent=2))
            continue

        test_case = LLMTestCase(input=entry["query"], actual_output=entry["answer"],
                                retrieval_context=entry["contexts"] or
                                ["No document chunks were retrieved."])
        scores, reasons = {}, {}
        try:
            fm = FaithfulnessMetric(threshold=0.6, model=judge, include_reason=True)
            fm.measure(test_case)
            scores["faithfulness"] = round(float(fm.score), 4)
            reasons["faithfulness"] = (getattr(fm, "reason", None) or "")[:400]
        except Exception as e:
            scores["faithfulness"] = None
            reasons["faithfulness"] = f"ERROR: {str(e)[:200]}"
        try:
            r_score, r_reason = _relevancy_judge(entry["query"], entry["answer"],
                                                 pinned_judge=judge if independent else None)
            scores["relevancy"] = round(r_score, 4)
            reasons["relevancy"] = r_reason
        except Exception as e:
            scores["relevancy"] = None
            reasons["relevancy"] = f"ERROR: {str(e)[:200]}"
        results[qid] = {"type": case["type"], **scores, "reasons": reasons}
        results_path.write_text(json.dumps(results, indent=2))
        print(f"[judge] {i+1}/{len(cases)} {qid} F={scores['faithfulness']} R={scores['relevancy']}", flush=True)

    remaining = sum(1 for c in cases if c["qid"] not in results)
    print(f"[judge] {len(results)} total graded -> {results_path.name}; "
          f"{remaining} deferred (answers not yet generated — re-run this stage)")


# ------------------------------------------------------------------- summary

def summarize():
    answers = json.loads(ANSWERS_PATH.read_text())
    judge = json.loads(JUDGE_PATH.read_text())
    vals = {"faithfulness": [], "relevancy": []}
    per_type = {}
    cov = []
    for qid, j in judge.items():
        if j.get("skipped"):
            continue
        for k in vals:
            if isinstance(j.get(k), (int, float)):
                vals[k].append(j[k])
        t = answers.get(qid, {}).get("type", "?")
        m = per_type.setdefault(t, {"faithfulness": [], "relevancy": []})
        for k in vals:
            if isinstance(j.get(k), (int, float)):
                m[k].append(j[k])
        v = answers.get(qid, {}).get("verification") or {}
        if isinstance(v.get("citation_coverage"), (int, float)):
            cov.append(v["citation_coverage"])

    total_chars = sum(a.get("answer_chars", 0) + a.get("context_chars", 0) for a in answers.values())
    est_cost = total_chars / 4 * (0.15 + 0.50) / 2 / 1e6  # crude blended in/out estimate

    summary = {
        "n_answers": len(answers),
        "n_judged": sum(1 for j in judge.values() if not j.get("skipped")),
        "mean_faithfulness": round(sum(vals["faithfulness"]) / len(vals["faithfulness"]), 4) if vals["faithfulness"] else None,
        "mean_relevancy": round(sum(vals["relevancy"]) / len(vals["relevancy"]), 4) if vals["relevancy"] else None,
        "pass_rate_0.6": {
            "faithfulness": round(sum(1 for v in vals["faithfulness"] if v >= 0.6) / len(vals["faithfulness"]), 4) if vals["faithfulness"] else None,
            "relevancy": round(sum(1 for v in vals["relevancy"] if v >= 0.6) / len(vals["relevancy"]), 4) if vals["relevancy"] else None,
        },
        "mean_citation_coverage": round(sum(cov) / len(cov), 4) if cov else None,
        "per_type": {t: {"faithfulness": round(sum(m["faithfulness"]) / len(m["faithfulness"]), 4),
                         "relevancy": round(sum(m["relevancy"]) / len(m["relevancy"]), 4)}
                     for t, m in sorted(per_type.items()) if m["faithfulness"]},
        "estimated_cost_usd_generation": round(est_cost, 3),
        "judge_model": "gate rotation head (z-ai/glm-5.3-flash)",
        "caveat": "generator and judge are the same model family; Gemini cross-check pending",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    kwargs = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    stage = kwargs.get("stage", "summarize")
    start = int(kwargs.get("start", 0))
    end = int(kwargs.get("end", 60))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if stage == "generate":
        stage_generate(start, end)
    elif stage == "judge":
        judge_model = os.getenv("JUDGE_MODEL", "").strip() or None
        stage_judge(start, end, judge_model=judge_model)
    elif stage == "summarize":
        summarize()


if __name__ == "__main__":
    main()
