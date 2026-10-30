"""Generates hand-review CSVs:
  - evaluation_suite/review/gold_review_30.csv  (30 of the 60 gold pairs, stratified)
  - evaluation_suite/review/answer_grading_20.csv (20 answers with judge scores)

Fill the verdict columns by hand; read_review_sheets.py then reports the gold
error rate and judge-vs-human agreement. Sheets regenerate idempotently —
filled verdicts in an existing sheet are preserved when the same qids are
regenerated.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
REVIEW = ROOT / "evaluation_suite" / "review"
GOLD_CSV = REVIEW / "gold_review_30.csv"
GRADE_CSV = REVIEW / "answer_grading_20.csv"


def main():
    REVIEW.mkdir(parents=True, exist_ok=True)
    gold = json.loads((ROOT / "evaluation_suite/dataset/gold_qa_dataset.json").read_text())
    answers = json.loads((ROOT / "evaluation_suite/evaluations/llm_judge/rag_answers.json").read_text())
    judge = json.loads((ROOT / "evaluation_suite/evaluations/llm_judge/judge_results.json").read_text())

    cases = []
    for aid, entry in gold.items():
        for qa in entry["qa_pairs"]:
            cases.append({"qid": qa["question_id"], "paper": entry["paper_title"],
                          "question": qa["question"], "expected_answer": qa["expected_answer"],
                          "evidence_page": qa["evidence"].get("page"), "type": qa["question_type"]})
    cases.sort(key=lambda c: c["qid"])
    stratified_30 = cases[:: max(1, len(cases) // 30)][:30]

    def load_filled(path, key_col):
        filled = {}
        if path.exists():
            with open(path) as f:
                for row in csv.DictReader(f):
                    filled[row[key_col]] = row
        return filled

    prev_gold = load_filled(GOLD_CSV, "qid")
    with open(GOLD_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "paper", "type", "evidence_page", "question", "expected_answer",
                    "verdict(correct/incorrect/unclear)", "corrected_answer", "notes"])
        for c in stratified_30:
            prev = prev_gold.get(c["qid"], {})
            w.writerow([c["qid"], c["paper"], c["type"], c["evidence_page"],
                        c["question"], c["expected_answer"],
                        prev.get("verdict(correct/incorrect/unclear)", ""),
                        prev.get("corrected_answer", ""), prev.get("notes", "")])

    qids = sorted(answers.keys())
    stratified_20 = qids[:: max(1, len(qids) // 20)][:20]
    prev_grade = load_filled(GRADE_CSV, "qid")
    with open(GRADE_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "question", "answer", "judge_faithfulness", "judge_relevancy",
                    "human_faithfulness_0_1", "human_relevancy_0_1", "notes"])
        for qid in stratified_20:
            e = answers.get(qid, {})
            j = judge.get(qid, {})
            prev = prev_grade.get(qid, {})
            w.writerow([qid, e.get("query", ""), e.get("answer", ""),
                        j.get("faithfulness", ""), j.get("relevancy", ""),
                        prev.get("human_faithfulness_0_1", ""), prev.get("human_relevancy_0_1", ""),
                        prev.get("notes", "")])

    print(f"[review] wrote {GOLD_CSV.name} ({len(stratified_30)} rows, "
          f"{sum(1 for c in stratified_30 if prev_gold.get(c['qid'], {}).get('verdict(correct/incorrect/unclear)'))} filled)")
    print(f"[review] wrote {GRADE_CSV.name} ({len(stratified_20)} rows)")


if __name__ == "__main__":
    main()
