"""
RAG vs ChatGPT Evaluation using deepeval + Gemini as LLM Judge
===============================================================
Loads BOTH:
  • evaluation_dataset.json  — ChatGPT-generated gold answers + evidence
  • rag_output_queries.json  — RAG system answers + retrieved chunks

For every question, Gemini (via deepeval) compares the RAG answer against
the gold answer and rates the RAG system on 5 standard RAG metrics.
Results are saved to evaluation_ragvschatgpt.json.

Metrics:
  • Faithfulness           — RAG answer grounded in retrieved chunks?
  • Answer Relevancy       — RAG answer on-topic for the question?
  • Contextual Precision   — Retrieved chunks ranked well for the question?
  • Contextual Recall      — Chunks cover what the gold answer needs?
  • Answer Correctness     — RAG answer matches the gold answer? (via GEval)

Usage:
    python3 evaluate_ragvschatgpt.py
"""

import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# ── Load env ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY not found in .env")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent
EVAL_DATASET = BASE / "gold_qa_dataset.json"
RAG_OUTPUT   = BASE / "rag_retrieved_answers.json"
RESULTS_FILE = BASE / "deep_eval_results.json"

# ── deepeval imports ──────────────────────────────────────────────────────────
try:
    from deepeval.models import GeminiModel
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        GEval,
    )
    from deepeval.test_case import LLMTestCaseParams
except ImportError as e:
    print(f"❌ deepeval not installed: {e}")
    print("   Run: pip install deepeval google-generativeai")
    sys.exit(1)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)

def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def build_test_cases(eval_dataset: dict, rag_output: dict) -> list:
    """
    Merge evaluation_dataset.json (gold) + rag_output_queries.json (RAG)
    into deepeval LLMTestCase objects — one per question_id.
    """
    cases   = []
    skipped = 0

    for paper_id, paper_data in eval_dataset.items():
        for qa in paper_data.get("qa_pairs", []):
            qid = qa["question_id"]

            if qid not in rag_output:
                skipped += 1
                continue

            rag_item = rag_output[qid]

            # Retrieval context = actual chunks retrieved by the RAG system
            retrieval_context = [
                c["text"]
                for c in rag_item.get("retrieved_chunks", [])
                if c.get("text", "").strip()
            ] or [""]

            test_case = LLMTestCase(
                input             = qa["question"],
                actual_output     = rag_item.get("rag_answer", ""),
                expected_output   = qa["expected_answer"],      # gold answer from ChatGPT
                retrieval_context = retrieval_context,
            )

            meta = {
                "question_id":     qid,
                "paper_title":     paper_data.get("paper_title", ""),
                "question_type":   qa.get("question_type", "text"),
                "difficulty":      qa.get("difficulty", ""),
                "question":        qa["question"],
                "expected_answer": qa["expected_answer"],
                "rag_answer":      rag_item.get("rag_answer", ""),
                "latency_sec":     rag_item.get("latency_sec", 0),
                "evidence":        qa.get("evidence", {}),
                "retrieved_chunks":rag_item.get("retrieved_chunks", []),
            }
            cases.append((qid, meta, test_case))

    if skipped:
        print(f"⚠️  {skipped} QA pairs skipped (not yet in rag_output_queries.json — run run_rag_evaluation.py first)")
    return cases

# ── Main ──────────────────────────────────────────────────────────────────────

import argparse

def run():
    parser = argparse.ArgumentParser(description="Evaluate RAG Pipeline using DeepEval")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test cases to evaluate")
    args = parser.parse_args()

    print("Loading datasets...")
    eval_dataset = load_json(EVAL_DATASET)
    rag_output   = load_json(RAG_OUTPUT)
    results      = load_json(RESULTS_FILE)

    if not eval_dataset:
        print(f"❌ {EVAL_DATASET} not found. Run the QA generation step first.")
        sys.exit(1)
    if not rag_output:
        print(f"❌ {RAG_OUTPUT} not found.")
        print("   Run:  python3 run_rag_evaluation.py  first.")
        sys.exit(1)

    # ── Initialise Gemini as judge ────────────────────────────────────────────
    print("Initialising Gemini judge (models/gemini-2.5-flash)...")
    judge = GeminiModel(
        model_name = "models/gemini-2.5-flash",
        api_key    = GEMINI_API_KEY,
    )

    # ── Define metrics ────────────────────────────────────────────────────────
    # GEval for answer correctness (compares RAG answer vs gold answer)
    correctness_metric = GEval(
        name       = "Answer Correctness",
        model      = judge,
        criteria   = (
            "Determine how closely the 'actual output' matches the 'expected output'. "
            "Award a high score if the key facts, figures, and reasoning are the same. "
            "Award a low score if the actual output contradicts, misses key points, or "
            "contains hallucinated information not in the expected output."
        ),
        evaluation_params = [
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold = 0.5,
    )

    metrics = {
        "faithfulness":         FaithfulnessMetric(       threshold=0.5, model=judge, include_reason=True),
        "answer_relevancy":     AnswerRelevancyMetric(    threshold=0.5, model=judge, include_reason=True),
        "contextual_precision": ContextualPrecisionMetric(threshold=0.5, model=judge, include_reason=True),
        "contextual_recall":    ContextualRecallMetric(   threshold=0.5, model=judge, include_reason=True),
        "contextual_relevancy": ContextualRelevancyMetric(threshold=0.5, model=judge, include_reason=True),
        "answer_correctness":   correctness_metric,
    }

    # ── Build & filter test cases ─────────────────────────────────────────────
    all_cases = build_test_cases(eval_dataset, rag_output)
    pending   = [(qid, meta, tc) for qid, meta, tc in all_cases if qid not in results]

    # Apply limit if specified
    if args.limit is not None:
        pending = pending[:args.limit]
        print(f"⚠️ Limit enabled: capping evaluation to the first {args.limit} pending cases.")

    total_matched = len(all_cases)
    print(f"\nQA pairs matched across both files : {total_matched}")
    print(f"Already evaluated                  : {total_matched - len([c for c in all_cases if c[0] not in [p[0] for p in pending] and c[0] not in results])}")
    print(f"Pending                            : {len(pending)}\n")

    if not pending:
        print("✅ All pairs already evaluated!")
        print_summary(results)
        return

    # ── Evaluate — one pair at a time, save after each ───────────────────────
    for i, (qid, meta, test_case) in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] {qid}  |  type={meta['question_type']}  |  {meta['question'][:80]}...")

        scores  = {}
        reasons = {}

        for name, metric in metrics.items():
            try:
                metric.measure(test_case)
                score = round(float(metric.score), 4)
                scores[name]  = score
                reasons[name] = getattr(metric, "reason", "") or ""
                tag = "✓" if score >= metric.threshold else "✗"
                print(f"    {tag} {name:<25} {score:.4f}")
            except Exception as e:
                print(f"    ⚠️ {name} error: {e}")
                scores[name]  = None
                reasons[name] = str(e)

            time.sleep(0.5)   # small pause between Gemini API calls per metric

        # Aggregate overall score
        valid   = [v for v in scores.values() if v is not None]
        overall = round(sum(valid) / len(valid), 4) if valid else None
        passed  = sum(1 for v in valid if v >= 0.5)

        print(f"    → Overall: {overall}  |  {passed}/{len(metrics)} metrics passed\n")

        results[qid] = {
            **meta,
            "scores":        scores,
            "reasons":       reasons,
            "overall_score": overall,
            "metrics_passed": passed,
            "metrics_total":  len(metrics),
        }

        save_json(RESULTS_FILE, results)

    print(f"✅ Evaluation complete!  {len(pending)} pairs evaluated.")
    print(f"   Results → {RESULTS_FILE}\n")
    print_summary(results)


def print_summary(results: dict):
    if not results:
        return

    metric_keys = [
        "faithfulness", "answer_relevancy", "contextual_precision",
        "contextual_recall", "contextual_relevancy", "answer_correctness",
    ]
    agg     = {m: [] for m in metric_keys}
    overall = []
    by_type = {}

    for r in results.values():
        sc = r.get("scores", {})
        for m in metric_keys:
            v = sc.get(m)
            if v is not None:
                try:
                    agg[m].append(float(v))
                except Exception:
                    pass
        ov = r.get("overall_score")
        if ov is not None:
            try:
                overall.append(float(ov))
            except Exception:
                pass
        qt = r.get("question_type", "unknown")
        if ov is not None:
            by_type.setdefault(qt, []).append(float(ov))

    print(f"\n{'='*62}")
    print(f"  RAG PIPELINE EVALUATION  ({len(results)} pairs — Gemini judge)")
    print(f"{'='*62}")
    for m in metric_keys:
        vals = agg[m]
        if vals:
            avg  = sum(vals) / len(vals)
            bar  = "█" * int(avg * 10) + "░" * (10 - int(avg * 10))
            label = m.replace("_", " ").title()
            print(f"  {label:<26}  {bar}  {avg:.3f}")
    if overall:
        avg = sum(overall) / len(overall)
        bar = "█" * int(avg * 10) + "░" * (10 - int(avg * 10))
        print(f"  {'Overall':<26}  {bar}  {avg:.3f}")
    print(f"{'='*62}")

    if by_type:
        print(f"\n  By question type (overall avg):")
        for qt, vals in sorted(by_type.items()):
            avg = sum(vals) / len(vals)
            print(f"    {qt:<14}  {avg:.3f}")
        print()

    # Pass rate
    total_pairs  = len(results)
    passed_pairs = sum(1 for r in results.values() if (r.get("overall_score") or 0) >= 0.5)
    print(f"  Pass rate (overall ≥ 0.5): {passed_pairs}/{total_pairs}  "
          f"({100*passed_pairs/total_pairs:.1f}%)")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    run()
