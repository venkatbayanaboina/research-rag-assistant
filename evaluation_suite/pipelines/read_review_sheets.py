"""Reads the hand-filled review sheets and reports:
  - gold error rate: fraction of reviewed pairs judged incorrect/unclear
  - judge-vs-human agreement: mean absolute difference and pass/fail agreement
    at the 0.6 threshold (both metrics, where the human filled scores)

Run after filling the verdict/score columns in
  evaluation_suite/review/gold_review_30.csv
  evaluation_suite/review/answer_grading_20.csv
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
GOLD_CSV = ROOT / "evaluation_suite/review/gold_review_30.csv"
GRADE_CSV = ROOT / "evaluation_suite/review/answer_grading_20.csv"


def main():
    # Gold error rate
    gold_rows = list(csv.DictReader(open(GOLD_CSV))) if GOLD_CSV.exists() else []
    reviewed = [r for r in gold_rows if r.get("verdict(correct/incorrect/unclear)", "").strip()]
    incorrect = [r for r in reviewed if r["verdict(correct/incorrect/unclear)"].strip().lower() == "incorrect"]
    unclear = [r for r in reviewed if r["verdict(correct/incorrect/unclear)"].strip().lower() == "unclear"]
    print(f"[gold] reviewed {len(reviewed)}/{len(gold_rows)} | "
          f"incorrect {len(incorrect)} | unclear {len(unclear)} | "
          f"error rate {len(incorrect)/len(reviewed)*100:.1f}%" if reviewed else "[gold] nothing filled")
    for r in incorrect:
        print(f"  incorrect: {r['qid']} — {r['notes'][:80]}")

    # Judge vs human agreement
    grade_rows = list(csv.DictReader(open(GRADE_CSV))) if GRADE_CSV.exists() else []
    diffs = {"faithfulness": [], "relevancy": []}
    agree = {"faithfulness": [0, 0], "relevancy": [0, 0]}  # [agreed, total]
    filled = 0
    for r in grade_rows:
        if (r.get("human_faithfulness_0_1", "").strip()
                or r.get("human_relevancy_0_1", "").strip()):
            filled += 1
        for metric in ("faithfulness", "relevancy"):
            try:
                human = float(r.get(f"human_{metric}_0_1", ""))
                judge = float(r.get(f"judge_{metric}", ""))
            except (TypeError, ValueError):
                continue
            diffs[metric].append(abs(human - judge))
            agree[metric][1] += 1
            if (human >= 0.6) == (judge >= 0.6):
                agree[metric][0] += 1

    print(f"[answers] graded rows present: {len(grade_rows)}, human scores filled: {filled}")
    for metric in ("faithfulness", "relevancy"):
        if diffs[metric]:
            print(f"  {metric}: mean |human-judge| = {sum(diffs[metric])/len(diffs[metric]):.3f} "
                  f"(n={len(diffs[metric])}) | pass/fail agreement "
                  f"{agree[metric][0]}/{agree[metric][1]}")
        else:
            print(f"  {metric}: no human scores filled yet")


if __name__ == "__main__":
    main()
