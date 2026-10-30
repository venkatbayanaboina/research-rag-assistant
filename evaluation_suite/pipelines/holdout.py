"""Hold-out evaluation set: schema, loader, and validator.

The hold-out set lives at eval/holdout.jsonl — ONE JSON object per line:
    {
      "qid": "H001",                          # unique
      "question": "...",                      # non-empty
      "gold_answer": "...",                   # non-empty when answerable
      "gold_pages": [1, 3],                   # list of ints (PDF page numbers)
      "gold_chunk_ids": [12, 47],             # list of ints; empty allowed
      "type": "text|equation|figure|table|comparison|other",
      "answerable": true                      # false = the set must abstain
    }

The file is intentionally EMPTY: questions are authored by a human, never
generated, and never used for any tuning decision. eval/FROZEN_COMMIT.txt
records the commit the hold-out set was validated against before it may be
evaluated against a different commit.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
HOLDOUT_PATH = ROOT / "eval" / "holdout.jsonl"
FROZEN_PATH = ROOT / "eval" / "FROZEN_COMMIT.txt"

VALID_TYPES = {"text", "equation", "figure", "table", "comparison", "other"}
REQUIRED = ("qid", "question", "gold_answer", "gold_pages", "gold_chunk_ids", "type", "answerable")
_HASH_RE = re.compile(r"^[0-9a-f]{40}$")


def load_holdout():
    """Returns the list of valid records ([] if the file is absent/empty)."""
    return validate_holdout(read_raw())


def read_raw():
    if not HOLDOUT_PATH.exists():
        return []
    records = []
    for line in HOLDOUT_PATH.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def validate_holdout(records):
    """Returns valid records; raises ValueError listing every problem found."""
    problems, seen = [], set()
    valid = []
    for i, r in enumerate(records, start=1):
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            problems.append(f"line {i}: missing fields {missing}")
            continue
        if r["qid"] in seen:
            problems.append(f"line {i}: duplicate qid {r['qid']}")
            continue
        if not isinstance(r["question"], str) or not r["question"].strip():
            problems.append(f"line {i} ({r['qid']}): empty question")
            continue
        if not isinstance(r["gold_pages"], list) or not all(isinstance(p, int) for p in r["gold_pages"]):
            problems.append(f"line {i} ({r['qid']}): gold_pages must be a list of ints")
            continue
        if not isinstance(r["gold_chunk_ids"], list) or not all(isinstance(c, int) for c in r["gold_chunk_ids"]):
            problems.append(f"line {i} ({r['qid']}): gold_chunk_ids must be a list of ints")
            continue
        if r["type"] not in VALID_TYPES:
            problems.append(f"line {i} ({r['qid']}): type must be one of {sorted(VALID_TYPES)}")
            continue
        if not isinstance(r["answerable"], bool):
            problems.append(f"line {i} ({r['qid']}): answerable must be a bool")
            continue
        if r["answerable"] and (not isinstance(r["gold_answer"], str) or not r["gold_answer"].strip()):
            problems.append(f"line {i} ({r['qid']}): answerable=true requires non-empty gold_answer")
            continue
        seen.add(r["qid"])
        valid.append(r)
    if problems:
        raise ValueError("holdout.jsonl validation failed:\n  " + "\n  ".join(problems))
    return valid


def frozen_commit():
    """Returns the recorded frozen hash, or None if absent/invalid."""
    if not FROZEN_PATH.exists():
        return None
    text = FROZEN_PATH.read_text().strip().lower()
    m = _HASH_RE.search(text)
    return m.group(0) if m else None
