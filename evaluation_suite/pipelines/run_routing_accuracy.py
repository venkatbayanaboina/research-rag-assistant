"""Intent-routing accuracy evaluation.

Runs a labeled query set through route_user_intent (1 LLM call per query) and
scores: intent accuracy, target-doc accuracy, and focus extraction for the
research intents. Includes adversarial cases (figure references that must NOT
route to research intents, untargeted research questions, "last paper").

Usage:
    rag/bin/python3 -m evaluation_suite.pipelines.run_routing_accuracy
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "evaluation_suite" / "evaluations" / "routing_accuracy.json"

ATTN = "attention-is-all-you-need-Paper.pdf"
CNN = "1408.5882_Convolutional_Neural_Networks_for_Sentence_Classification.pdf"
S2S = "1409.3215_Sequence_to_Sequence_Learning_with_Neural_Networks.pdf"
AE = "1407.7906_How_Auto-Encoders_Could_Provide_Credit_Assignment_in_Deep_Networks_via_Target_Propagation.pdf"
SOC = "Social-Engineering-Attack.pdf"
SPOOF = "Spoofing_attack_using_bus-off_attacks_against_a_specific_ECU_of_the_CAN_bus.pdf"
EMO = "Theories-Methods-and-Current-Research-on-Emotions.pdf"
NET = "Computer-Networks-Deep-Dive (1).pdf"
LAST = sorted([ATTN, CNN, S2S, AE, SOC, SPOOF, EMO, NET])[-1]  # what the router's list calls "the last paper"

LABELED_QUERIES = [
    # STANDARD_CHAT — including the adversarial figure/table references
    ("what is self attention?", "STANDARD_CHAT", []),
    ("show me table 3", "STANDARD_CHAT", []),
    ("what is figure 2 about in the attention paper?", "STANDARD_CHAT", [ATTN]),
    ("explain the CMRR formula", "STANDARD_CHAT", []),
    ("hello there!", "STANDARD_CHAT", []),
    # SUMMARY
    ("summarize attention-is-all-you-need-Paper.pdf", "SUMMARY", [ATTN]),
    ("give me a summary of the last paper", "SUMMARY", [LAST]),
    ("summarize the CNN paper", "SUMMARY", [CNN]),
    # SECTION_SUMMARY
    ("give me a section-wise breakdown of the attention paper", "SECTION_SUMMARY", [ATTN]),
    ("detailed section summary of Social-Engineering-Attack.pdf", "SECTION_SUMMARY", [SOC]),
    # COMPARISON
    ("compare the attention paper and the social engineering attack paper", "COMPARISON", [ATTN, SOC]),
    ("how does the transformer differ from the seq2seq paper?", "COMPARISON", [ATTN, S2S]),
    # EVIDENCE_TABLE (targeted, so the intent is executable)
    ("build an evidence table of method, dataset and results for the attention and CNN papers",
     "EVIDENCE_TABLE", [ATTN, CNN]),
    ("create an evidence table comparing the seq2seq and autoencoder papers",
     "EVIDENCE_TABLE", [S2S, AE]),
    ("what method, dataset and results do the emotions and spoofing papers report?",
     "EVIDENCE_TABLE", [EMO, SPOOF]),
    # CONTRADICTION
    ("do the attention and CNN papers contradict each other on accuracy?",
     "CONTRADICTION", [ATTN, CNN]),
    ("are there contradictions between the computer networks, social engineering and spoofing papers about security?",
     "CONTRADICTION", [NET, SOC, SPOOF]),
    ("does the seq2seq paper conflict with the autoencoder paper on training methods?",
     "CONTRADICTION", [S2S, AE]),
    # GAP_EXPLORATION
    ("what research gaps exist in the attention and emotions papers?",
     "GAP_EXPLORATION", [ATTN, EMO]),
    ("explore open problems in the social engineering and spoofing papers",
     "GAP_EXPLORATION", [SOC, SPOOF]),
    ("what has not been studied in the CNN and seq2seq papers?",
     "GAP_EXPLORATION", [CNN, S2S]),
]


def main():
    sys.path.insert(0, str(ROOT))
    from src.core.generator.router import route_user_intent
    from src.core.vector_store import get_indexed_documents

    indexed = get_indexed_documents()
    print(f"[routing] {len(LABELED_QUERIES)} labeled queries, {len(indexed)} indexed docs")

    results = []
    intent_ok = target_ok = 0
    per_intent = {}
    for i, (query, exp_intent, exp_targets) in enumerate(LABELED_QUERIES):
        t0 = time.time()
        got = route_user_intent(query, indexed)
        wall = time.time() - t0
        gi, gt = got.get("intent"), set(got.get("target_docs", []))
        et = set(exp_targets)
        i_ok = gi == exp_intent
        t_ok = gt == et if et else True  # empty expectation = no targets required
        intent_ok += i_ok
        target_ok += (i_ok and t_ok)
        m = per_intent.setdefault(exp_intent, {"n": 0, "intent_ok": 0, "target_ok": 0})
        m["n"] += 1
        m["intent_ok"] += i_ok
        m["target_ok"] += (i_ok and t_ok)
        flag = "OK " if (i_ok and t_ok) else "MISS"
        print(f"[routing] {flag} {query[:55]:55s} -> {gi} targets={sorted(gt)} ({wall:.1f}s)")
        results.append({"query": query, "expected_intent": exp_intent, "expected_targets": exp_targets,
                        "got_intent": gi, "got_targets": sorted(gt), "focus": got.get("focus", ""),
                        "intent_ok": i_ok, "target_ok": i_ok and t_ok, "wall_seconds": round(wall, 2)})
        time.sleep(0.5)

    n = len(results)
    summary = {
        "n_queries": n,
        "intent_accuracy": round(intent_ok / n, 4),
        "target_accuracy": round(target_ok / n, 4),
        "per_intent": {k: {"n": v["n"],
                           "intent_accuracy": round(v["intent_ok"] / v["n"], 4),
                           "target_accuracy": round(v["target_ok"] / v["n"], 4)}
                       for k, v in sorted(per_intent.items())},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"\n[routing] intent accuracy: {summary['intent_accuracy']*100:.1f}%  "
          f"target accuracy: {summary['target_accuracy']*100:.1f}%  saved -> {OUT.name}")


if __name__ == "__main__":
    main()
