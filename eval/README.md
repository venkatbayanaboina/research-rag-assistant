# Hold-out Evaluation Set

**Status: EMPTY — questions are human-authored by design.** The pipeline never
generates hold-out questions or gold answers, and nothing in this directory may
be used for tuning.

## Files

- `papers/` — the four original paper PDFs (byte-identical copies from
  `evaluation_suite/configs/pdfs/`, sha256-verified at copy time) to review the
  hold-out questions against:
  - `1407.7906_How_Auto-Encoders_…Target_Propagation.pdf` (H001–H008)
  - `1408.5882_Convolutional_Neural_Networks_for_Sentence_Classification.pdf` (H009–H016)
  - `1409.3215_Sequence_to_Sequence_Learning_with_Neural_Networks.pdf` (H017–H024)
  - `1409.7770_Entanglement-Based_Machine_Learning_on_a_Quantum_Computer.pdf` (H025–H032)
- `holdout_review_32.csv` is also at `evaluation_suite/review/holdout_review_32.csv`
  (same folder as the other review sheets) — that is the sheet to fill.
- `holdout.jsonl` — one JSON object per line (**currently 0 records**):
  ```json
  {"qid": "H001", "question": "...", "gold_answer": "...", "gold_pages": [1, 3],
   "gold_chunk_ids": [12, 47], "type": "text|equation|figure|table|comparison|other",
   "answerable": true}
  ```
- `FROZEN_COMMIT.txt` — the 40-hex commit hash the set was validated against
  (create it before evaluating; the runner refuses without it).

## Workflow

1. Hand-write questions into `holdout.jsonl` (schema validated by
   `evaluation_suite/pipelines/holdout.py::validate_holdout`).
2. `git rev-parse HEAD > eval/FROZEN_COMMIT.txt`
3. `make eval` will NOT touch this set; run it explicitly:
   `rag/bin/python3 -m evaluation_suite.pipelines.run_holdout_eval`
   The runner **refuses** unless the set is valid and the commit is frozen
   (both refusals verified — see docs/upgrade/WORK_RESULTS.md §9b).
4. Grades: retrieval vs gold_pages/gold_chunk_ids, answerable-vs-abstention
   behavior; results in `evaluation_suite/evaluations/holdout_results.json`.

## Why the refusal gates exist

The integrity audit (docs/upgrade/evaluation_results.md §8) found tuning-on-test
contamination: settings were adopted based on dev-set scores. The refusal gates
make it impossible to repeat that with the hold-out set.
