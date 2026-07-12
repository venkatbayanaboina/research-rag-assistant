"""
llm_judge.py
-------------
Runs LLM-as-Judge evaluation on sample_100_answers.json.

For each question it scores the RAG answer vs the gold expected_answer
on three dimensions:
  • Correctness   (1–5): Is the factual content correct?
  • Completeness  (1–5): Does it cover all key points?
  • Relevance     (1–5): Is it focused and on-topic?

Saves per-question scores + aggregate stats to:
  papers download/llm_judge_results.json
  papers download/llm_judge_report.txt
"""

import os
import sys
import json
import time
import requests
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
SAMPLE_IN   = ROOT / "papers download" / "sample_100_answers.json"
JUDGE_OUT   = ROOT / "papers download" / "llm_judge_results.json"
REPORT_OUT  = ROOT / "papers download" / "llm_judge_report.txt"

# ── Cerebras API ──────────────────────────────────────────────────────────────
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "CEREBRAS_API_KEY")
API_URL  = "https://api.cerebras.ai/v1/chat/completions"
MODEL    = "gpt-oss-120b"
RATE_LIMIT_DELAY = 6.0

JUDGE_PROMPT_TEMPLATE = """\
You are an expert evaluator assessing the quality of an AI-generated answer to a research question.

=== QUESTION ===
{question}

=== GOLD ANSWER (Expected) ===
{expected_answer}

=== RAG-GENERATED ANSWER ===
{rag_answer}

=== EVALUATION TASK ===
Score the RAG-Generated Answer on these THREE dimensions compared to the Gold Answer.
Use a scale of 1 to 5 for each:

- Correctness  : Is the factual content accurate and consistent with the Gold Answer? (1=wrong, 5=fully correct)
- Completeness : Does it cover all key points from the Gold Answer? (1=missing most, 5=covers everything)
- Relevance    : Is it focused on the question and free of irrelevant content? (1=off-topic, 5=perfectly focused)

IMPORTANT: Respond ONLY with valid JSON in this exact format, nothing else:
{{
  "correctness": <1-5>,
  "completeness": <1-5>,
  "relevance": <1-5>,
  "reasoning": "<one sentence explaining your scores>"
}}
"""

def load_sample() -> dict:
    if not SAMPLE_IN.exists():
        print(f"❌ {SAMPLE_IN} not found. Run sample_and_generate.py first.")
        sys.exit(1)
    with open(SAMPLE_IN) as f:
        return json.load(f)

def load_results() -> dict:
    if JUDGE_OUT.exists():
        with open(JUDGE_OUT) as f:
            return json.load(f)
    return {}

def save_results(data: dict):
    temp = JUDGE_OUT.with_suffix(".tmp")
    with open(temp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(temp, JUDGE_OUT)

def query_cerebras(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    backoff = 5
    for _ in range(6):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            elif r.status_code == 429:
                print(f"  ⚠️  Rate limit. Backing off {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
            else:
                print(f"  ⚠️  HTTP {r.status_code}: {r.text[:100]}")
                time.sleep(3)
        except Exception as e:
            print(f"  ⚠️  Exception: {e}")
            time.sleep(3)
    return ""

def parse_scores(text: str) -> dict:
    """Extract JSON scores from LLM output, robust to markdown fences."""
    import re
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        data = json.loads(text)
        return {
            "correctness":  int(data.get("correctness", 0)),
            "completeness": int(data.get("completeness", 0)),
            "relevance":    int(data.get("relevance", 0)),
            "reasoning":    str(data.get("reasoning", "")),
        }
    except Exception:
        return {"correctness": 0, "completeness": 0, "relevance": 0, "reasoning": "parse_error"}

def print_report(results: dict):
    scored = [v for v in results.values() if v.get("correctness", 0) > 0]
    if not scored:
        print("No scored entries yet.")
        return

    avg_c  = sum(v["correctness"]  for v in scored) / len(scored)
    avg_co = sum(v["completeness"] for v in scored) / len(scored)
    avg_r  = sum(v["relevance"]    for v in scored) / len(scored)
    avg    = (avg_c + avg_co + avg_r) / 3

    # Difficulty breakdown
    by_diff = {}
    for v in scored:
        d = v.get("difficulty", "unknown")
        by_diff.setdefault(d, []).append((v["correctness"] + v["completeness"] + v["relevance"]) / 3)

    lines = [
        "=" * 60,
        "  LLM-as-Judge Evaluation Report",
        "=" * 60,
        f"  Evaluated:      {len(scored)} questions",
        f"  Avg Correctness:   {avg_c:.2f} / 5",
        f"  Avg Completeness:  {avg_co:.2f} / 5",
        f"  Avg Relevance:     {avg_r:.2f} / 5",
        f"  ─────────────────────────────",
        f"  OVERALL AVG:       {avg:.2f} / 5  ({avg/5*100:.1f}%)",
        "",
        "  Breakdown by Difficulty:",
    ]
    for diff, scores in sorted(by_diff.items()):
        lines.append(f"    {diff:10s}: {sum(scores)/len(scores):.2f} / 5  (n={len(scores)})")

    # Question-type breakdown
    by_type = {}
    for v in scored:
        t = v.get("question_type", "unknown")
        by_type.setdefault(t, []).append((v["correctness"] + v["completeness"] + v["relevance"]) / 3)
    lines.append("")
    lines.append("  Breakdown by Question Type:")
    for qtype, scores in sorted(by_type.items()):
        lines.append(f"    {qtype:12s}: {sum(scores)/len(scores):.2f} / 5  (n={len(scores)})")

    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    with open(REPORT_OUT, "w") as f:
        f.write(report + "\n")
    print(f"\n📄  Full report saved to: {REPORT_OUT}")

def main():
    print("=" * 60)
    print("⚖️   Step 2: LLM-as-Judge Evaluation")
    print("=" * 60)

    sample = load_sample()
    results = load_results()

    # Find entries that have answers but haven't been judged yet
    to_judge = [
        qid for qid, item in sample.items()
        if item.get("rag_answer", "").strip()
        and not item.get("rag_answer", "").startswith("[ERROR")
        and qid not in results
    ]

    skipped = [qid for qid in sample if not sample[qid].get("rag_answer","").strip()]

    print(f"📋  Sample size:   {len(sample)}")
    print(f"✅  Already judged: {len(results)}")
    print(f"⏳  To judge:       {len(to_judge)}")
    print(f"⚠️   Skipped (no answer): {len(skipped)}")
    print(f"⏱️   Estimated time: ~{len(to_judge) * RATE_LIMIT_DELAY / 60:.1f} minutes")
    print()

    if not to_judge:
        print("✅  All judged! Printing report...")
        print_report(results)
        return

    for idx, qid in enumerate(to_judge, 1):
        item = sample[qid]
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=item["question"],
            expected_answer=item.get("expected_answer", "N/A"),
            rag_answer=item.get("rag_answer", ""),
        )

        raw = query_cerebras(prompt)
        scores = parse_scores(raw)

        results[qid] = {
            "question_id":     qid,
            "question":        item["question"],
            "difficulty":      item.get("difficulty", ""),
            "question_type":   item.get("question_type", ""),
            "expected_answer": item.get("expected_answer", ""),
            "rag_answer":      item.get("rag_answer", ""),
            **scores,
        }
        save_results(results)

        avg = (scores["correctness"] + scores["completeness"] + scores["relevance"]) / 3
        print(f"  ⚖️  [{idx}/{len(to_judge)}] {qid} | C={scores['correctness']} Co={scores['completeness']} R={scores['relevance']} | avg={avg:.1f}")

        time.sleep(RATE_LIMIT_DELAY)

    print()
    print("🎉  Judging complete!")
    print_report(results)

if __name__ == "__main__":
    main()
