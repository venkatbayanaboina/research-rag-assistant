import os
import sys
import json
import time
from pathlib import Path

# Insert root directory for imports
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.core.vector_store import get_registry
from src.core.generator.summarization import generate_summary
from src.core.generator.comparison import generate_comparison
from src.core.generator.client import generate_content_with_retry

# Define evaluation directories
AUDIT_DIR = ROOT / "evaluation_suite" / "evaluations" / "features_audit"
SUMMARIES_DIR = AUDIT_DIR / "summaries"
COMPARISONS_DIR = AUDIT_DIR / "comparisons"

os.makedirs(SUMMARIES_DIR, exist_ok=True)
os.makedirs(COMPARISONS_DIR, exist_ok=True)

# Select 5 target papers currently indexed in the FAISS registry
TARGET_PAPERS = [
    "attention-is-all-you-need-Paper.pdf",
    "1408.5882_Convolutional_Neural_Networks_for_Sentence_Classification.pdf",
    "1407.7906_How_Auto-Encoders_Could_Provide_Credit_Assignment_in_Deep_Networks_via_Target_Propagation.pdf",
    "1409.3215_Sequence_to_Sequence_Learning_with_Neural_Networks.pdf",
    "Social-Engineering-Attack.pdf"
]

def run_feature_generation_and_evaluation():
    print("=" * 70)
    print("🔬 Starting Multimodal Feature Audit Evaluation (5 Target Papers)")
    print("=" * 70)
    
    registry = get_registry()
    if not registry:
        print("❌ Registry is empty or chunks.json not found in storage. Ingestion must run first.")
        sys.exit(1)
        
    # 1. Generate Summaries for all 5 papers
    summaries_results = {}
    for doc_name in TARGET_PAPERS:
        doc_chunks = [chunk for chunk in registry if chunk["source_file"] == doc_name]
        if not doc_chunks:
            print(f"⚠️ Chunks for '{doc_name}' not found in registry. Skipping.")
            continue
            
        print(f"📝 Generating summary for: {doc_name}...")
        summary = generate_summary(doc_chunks)
        
        # Save to disk
        summary_filename = f"{os.path.splitext(doc_name)[0]}_summary.md"
        with open(SUMMARIES_DIR / summary_filename, "w") as f:
            f.write(summary)
            
        summaries_results[doc_name] = {
            "summary_path": str(SUMMARIES_DIR / summary_filename),
            "summary_content": summary
        }
        print(f"  ✓ Summary saved to: {summary_filename}")
        time.sleep(2) # rate limit prevention

    if not summaries_results:
        print("❌ No summaries could be generated. Check your document registry.")
        sys.exit(1)

    # 2. Generate side-by-side comparison for all 5 papers
    print("\n📊 Generating dynamic 5-paper comparative analysis table...")
    docs_data = []
    for doc_name in TARGET_PAPERS:
        doc_chunks = [chunk for chunk in registry if chunk["source_file"] == doc_name]
        if doc_chunks:
            docs_data.append({"name": doc_name, "chunks": doc_chunks})
            
    comparison_content = generate_comparison(docs_data)
    comparison_filename = "5_Papers_Comparative_Analysis_Matrix.md"
    with open(COMPARISONS_DIR / comparison_filename, "w") as f:
        f.write(comparison_content)
    print(f"  ✓ Comparative matrix saved to: {comparison_filename}")
        
    print("\n⚖️ Evaluating generated features using LLM-as-Judge (Gemini)...")
    evaluation_logs = []
    
    # Judge Summaries
    for doc_name, data in summaries_results.items():
        judge_prompt = f"""
You are a senior scientific evaluator grading the quality of a generated research paper summary.
Determine if the summary is comprehensive, well-structured, factually consistent with academic standards, and covers the core architecture, key contributions, methodology, results, and equations.

Rate the summary on a scale of 0.0 to 100.0%.

SUMMARY CONTENT TO GRADE:
{data["summary_content"]}

Provide your output ONLY as a valid JSON object in this exact schema, nothing else:
{{
  "score": <score_as_float>,
  "reasoning": "<one sentence explanation of the score, notes, and specific technical elements found>"
}}
"""
        try:
            response = generate_content_with_retry(judge_prompt, temperature=0.0)
            # Parse JSON
            clean_res = response.strip()
            if "```" in clean_res:
                clean_res = clean_res.replace("```json", "").replace("```", "").strip()
            grade_data = json.loads(clean_res)
        except Exception as e:
            # Never fabricate a score on judge failure (audit defect D4): record
            # the failure explicitly so report averages stay trustworthy.
            grade_data = {"score": None, "reasoning": f"UNGRADED - judge failed: {e}"}

        evaluation_logs.append({
            "feature": f"Summary_{os.path.splitext(doc_name)[0]}_summary.md",
            "score": grade_data.get("score"),
            "reasoning": grade_data.get("reasoning", "")
        })
        print(f"  {'✓' if grade_data.get('score') is not None else '✗'} Graded summary of {doc_name}: {grade_data.get('score')}")
        time.sleep(2)
        
    # Judge Comparison
    judge_comp_prompt = f"""
You are a senior scientific evaluator grading the quality of a generated side-by-side paper comparison.
Verify if the comparison table covers key dimensions (methodology, key formulas, architecture, results, limitations) side-by-side with equal depth for all compared papers.

Rate the comparison on a scale of 0.0 to 100.0%.

COMPARISON CONTENT TO GRADE:
{comparison_content}

Provide your output ONLY as a valid JSON object in this exact schema, nothing else:
{{
  "score": <score_as_float>,
  "reasoning": "<one sentence explanation of the score and trade-offs highlighted>"
}}
"""
    try:
        response_comp = generate_content_with_retry(judge_comp_prompt, temperature=0.0)
        clean_res_comp = response_comp.strip()
        if "```" in clean_res_comp:
            clean_res_comp = clean_res_comp.replace("```json", "").replace("```", "").strip()
        grade_data_comp = json.loads(clean_res_comp)
    except Exception as e:
        grade_data_comp = {"score": None, "reasoning": f"UNGRADED - judge failed: {e}"}

    evaluation_logs.append({
        "feature": f"Comparison_{comparison_filename}",
        "score": grade_data_comp.get("score"),
        "reasoning": grade_data_comp.get("reasoning", "")
    })
    print(f"  {'✓' if grade_data_comp.get('score') is not None else '✗'} Graded comparison matrix: {grade_data_comp.get('score')}")
    
    # 4. Generate the final DeepEval Multimodal Features Report markdown file
    report_lines = [
        "# 🏆 DeepEval Multimodal Features Quality Report",
        "",
        "This report logs the quality evaluation scores for your summarizations and comparisons.",
        "",
        "## 📈 Performance Summary Table",
        "",
        "| Feature | Score | Reasoning / Notes |",
        "| :--- | :---: | :--- |"
    ]
    
    total_score = 0.0
    sum_score = 0.0
    sum_count = 0
    comp_score = 0.0
    comp_count = 0
    
    graded_total = 0
    for log in evaluation_logs:
        score_val = log["score"]
        score_str = f"{score_val:.2f}%" if isinstance(score_val, (int, float)) else "UNGRADED"
        report_lines.append(f"| **{log['feature']}** | **{score_str}** | {log['reasoning']} |")
        if not isinstance(score_val, (int, float)):
            continue
        graded_total += 1
        total_score += score_val
        if "Summary" in log["feature"]:
            sum_score += score_val
            sum_count += 1
        elif "Comparison" in log["feature"]:
            comp_score += score_val
            comp_count += 1

    avg_sum = sum_score / sum_count if sum_count else 0.0
    avg_comp = comp_score / comp_count if comp_count else 0.0
    overall = total_score / graded_total if graded_total else 0.0

    ungraded = len(evaluation_logs) - graded_total
    report_lines.extend([
        "",
        "## 📊 Feature Categories Averages",
        f"- **Average Summary Score:** **{avg_sum:.2f}%** (n={sum_count})",
        f"- **Average Comparison Score:** **{avg_comp:.2f}%** (n={comp_count})",
        f"- **OVERALL FEATURE SCORE:** **{overall:.2f}%** (n={graded_total} graded"
        + (f", {ungraded} UNGRADED — judge failures excluded from averages" if ungraded else "") + ")",
        ""
    ])
    
    report_content = "\n".join(report_lines)
    report_file_path = AUDIT_DIR / "deepeval_multimodal_features_report.md"
    with open(report_file_path, "w") as f:
        f.write(report_content)
        
    print("\n" + "="*70)
    print(f"🎉 Audit complete! Packaged results stored successfully:")
    print(f"📂 Summaries  : {SUMMARIES_DIR}")
    print(f"📂 Comparisons: {COMPARISONS_DIR}")
    print(f"📄 Report File: {report_file_path}")
    print("="*70)

if __name__ == "__main__":
    run_feature_generation_and_evaluation()
