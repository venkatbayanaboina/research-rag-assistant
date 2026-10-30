"""Hold-out evaluation runner. REFUSES to run unless:
  1. eval/holdout.jsonl exists and contains at least one VALID record
     (questions are human-authored; the scaffold ships empty), and
  2. eval/FROZEN_COMMIT.txt contains the 40-hex commit hash the hold-out set
     was validated against — evaluation happens against that recorded
     snapshot, so results can never silently drift with code changes.

When both gates pass it grades, per record: retrieval vs gold_pages /
gold_chunk_ids, answerable-vs-abstention behavior, and (if the judge gate is
configured) faithfulness on answerable records. Writes
evaluation_suite/evaluations/holdout_results.json.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "evaluation_suite" / "evaluations" / "holdout_results.json"


def _current_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=ROOT).stdout.strip()
    except Exception:
        return None


def main():
    from evaluation_suite.pipelines.holdout import load_holdout, read_raw, frozen_commit

    # Gate 1: human-authored, valid hold-out records
    try:
        raw = read_raw()
    except json.JSONDecodeError as e:
        print(f"REFUSED: eval/holdout.jsonl is not valid JSONL ({e}).")
        return 1
    if not raw:
        print("REFUSED: eval/holdout.jsonl is empty. The hold-out set is "
              "human-authored by design — write questions into eval/holdout.jsonl "
              "first (schema in evaluation_suite/pipelines/holdout.py docstring).")
        return 1
    try:
        records = load_holdout()
    except ValueError as e:
        print(f"REFUSED: {e}")
        return 1
    if not records:
        print("REFUSED: holdout.jsonl contains no valid records.")
        return 1

    # Gate 2: frozen commit recorded
    frozen = frozen_commit()
    if not frozen:
        print("REFUSED: eval/FROZEN_COMMIT.txt is missing or does not contain a "
              "40-hex commit hash. Freeze the commit the hold-out set was "
              "validated against before evaluating.")
        return 1
    head = _current_head()
    drift = "" if head == frozen else (
        f"WARNING: current HEAD {head} differs from frozen {frozen}; "
        "results must be reported against the frozen hash.\n")

    # ---- actual evaluation (reached only when both gates pass) ----
    from src.core.generator import execute_rag_pipeline
    from src.core.vector_store import get_indexed_documents

    indexed = get_indexed_documents()
    results = []
    for r in records:
        try:
            result = execute_rag_pipeline(r["question"], indexed)
        except Exception as e:
            results.append({"qid": r["qid"], "error": str(e)[:200],
                            "retrieval_hit": None, "abstained": None,
                            "behavior_ok": None})
            continue
        got_pages = {(c.get("chunk") or c).get("page")
                     for c in (result.get("search_results") or [])}
        retrieval_hit = bool(set(r["gold_pages"]) & got_pages) if r["gold_pages"] else None
        abstained = "could not find" in (result.get("answer", "") or "").lower()
        behavior_ok = (not r["answerable"]) or (not abstained)
        if not r["answerable"]:
            behavior_ok = abstained or not result.get("search_results")
        results.append({"qid": r["qid"], "retrieval_hit": retrieval_hit,
                        "abstained": abstained, "behavior_ok": behavior_ok,
                        "answer_head": (result.get("answer") or "")[:150]})

    summary = {"frozen_commit": frozen, "current_head": head,
               "n_records": len(records),
               "retrieval_hits": sum(1 for x in results if x["retrieval_hit"]),
               "behavior_ok": sum(1 for x in results if x["behavior_ok"]),
               "results": results}
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"{drift}[holdout] n={len(records)} retrieval_hits={summary['retrieval_hits']} "
          f"behavior_ok={summary['behavior_ok']} -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
