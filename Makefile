# Reproducible evaluation entry points.
# All numbers in docs/ are regenerated from raw JSON artifacts — nothing hand-typed.

PYTHON ?= rag/bin/python3
PIP = $(PYTHON) -m pip

.PHONY: test eval eval-live smoke review-sheets bootstrap lock secret-scan ci

test:            ## run the full offline unit suite (57 tests)
	$(PYTHON) -m pytest tests/test_fusion.py tests/test_sparse_retrieval.py \
		tests/test_verification.py tests/test_graph.py tests/test_defect_fixes.py \
		tests/test_features.py tests/test_visual_ranking.py tests/test_research_actions.py -q

smoke:           ## tiny offline smoke checks for CI
	$(PYTHON) -m evaluation_suite.pipelines.smoke_eval

bootstrap:       ## recompute bootstrap CIs from raw benchmark + judge JSONs
	$(PYTHON) -m evaluation_suite.pipelines.run_bootstrap_ci

eval: bootstrap  ## regenerate every reported number FROM RAW RESULT FILES (offline, no API)
	$(PYTHON) -m evaluation_suite.pipelines.run_failure_analysis
	$(PYTHON) -m evaluation_suite.pipelines.run_latency_report
	$(PYTHON) -m evaluation_suite.pipelines.render_performance_report
	$(PYTHON) -m evaluation_suite.pipelines.run_full_evaluation stage=summarize

eval-live:       ## re-run the LIVE stages (costs API credits, ~40-60 min)
	$(PYTHON) -m evaluation_suite.pipelines.run_retrieval_benchmark --mode dense  --k 5
	$(PYTHON) -m evaluation_suite.pipelines.run_retrieval_benchmark --mode hybrid --k 5
	$(PYTHON) -m evaluation_suite.pipelines.run_full_evaluation stage=generate start=0 end=60
	$(PYTHON) -m evaluation_suite.pipelines.run_full_evaluation stage=judge start=0 end=60
	$(PYTHON) -m evaluation_suite.pipelines.run_full_evaluation stage=summarize
	$(PYTHON) -m evaluation_suite.pipelines.run_routing_accuracy
	$(PYTHON) -m evaluation_suite.pipelines.run_gemini_crosscheck --n 20

review-sheets:   ## regenerate the hand-review CSVs (30 gold, 20 answers)
	$(PYTHON) -m evaluation_suite.pipelines.generate_review_sheets

lock:            ## re-pin requirements.lock.txt from the rag venv
	$(PIP) freeze --exclude-editable > requirements.lock.txt

secret-scan:     ## scan full git history for credential patterns
	$(PYTHON) scripts/secret_scan.py

ci: test smoke   ## what CI runs
