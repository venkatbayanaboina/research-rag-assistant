import os
import json
import time
import re
import torch
from pathlib import Path
from transformers import pipeline
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# ── 1. Auto-Detect Correct Paths ─────────────────────────────────────────────
if Path("research-rag-assistant/storage/text_db.faiss").exists():
    BASE_PATH = Path("research-rag-assistant")
else:
    BASE_PATH = Path(".")

ANSWERS_FILE = BASE_PATH / "storage" / "rag_retrieved_answers.json"
EVAL_OUT_FILE = BASE_PATH / "storage" / "deepeval_evaluation_results.json"
REPORT_FILE = BASE_PATH / "storage" / "deepeval_report.md"

print(f"📂 Selected active directory path: {BASE_PATH.resolve()}")
print(f"📄 Looking for answers file at: {ANSWERS_FILE}")

# Ensure storage directory exists
os.makedirs(ANSWERS_FILE.parent, exist_ok=True)

# ── 2. Create Local Qwen Evaluator Wrapper for DeepEval ─────────────────────
class LocalQwenEvaluator(DeepEvalBaseLLM):
    def __init__(self):
        print("Loading Qwen 2.5 3B onto GPU as evaluator...")
        self.pipe = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-3B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.pipe.tokenizer.pad_token_id = self.pipe.tokenizer.eos_token_id
        self.pipe.tokenizer.padding_side = "left"

    def load_model(self):
        return self.pipe

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        res = self.pipe(formatted_prompt, max_new_tokens=256, temperature=0.1, do_sample=False)
        return res[0]["generated_text"][len(formatted_prompt):].strip()

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Qwen 2.5 3B Instruct"

# Instantiate evaluator
eval_model = LocalQwenEvaluator()

# ── 3. Instantiate Metrics ──────────────────────────────────────────────────
faithfulness_metric = FaithfulnessMetric(threshold=0.6, model=eval_model)
relevancy_metric = AnswerRelevancyMetric(threshold=0.6, model=eval_model)

# ── 4. Load Answers and Filter ───────────────────────────────────────────────
with open(ANSWERS_FILE) as f:
    data = json.load(f)

# Select only questions that have generated answers
eval_items = []
for qid, item in data.items():
    ans = item.get("rag_answer", "").strip()
    if ans and not ans.startswith("[ERROR"):
        eval_items.append((qid, item))

print(f"Loaded {len(eval_items)} answered questions for evaluation.")

# Load previous results to resume
eval_results = {}
if EVAL_OUT_FILE.exists():
    try:
        with open(EVAL_OUT_FILE) as f:
            eval_results = json.load(f)
    except: pass

# Clean up failed runs to re-evaluate them
failed_qids = [qid for qid, v in eval_results.items() if v.get("status", "").startswith("FAILED")]
for qid in failed_qids:
    del eval_results[qid]

to_evaluate = [(qid, item) for qid, item in eval_items if qid not in eval_results]
print(f"Already evaluated: {len(eval_results)} | Pending: {len(to_evaluate)}")

# ── 5. Evaluation Loop ───────────────────────────────────────────────────────
t_start = time.time()

for idx, (qid, item) in enumerate(to_evaluate, start=1):
    context_str = "\n".join([f"- {c['text']}" for c in item.get("retrieved_chunks", [])[:3]])
    
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=item["rag_answer"],
        retrieval_context=[context_str]
    )
    
    t_case_start = time.time()
    try:
        # Measure Faithfulness
        faithfulness_metric.measure(test_case)
        f_score = faithfulness_metric.score
        f_reason = faithfulness_metric.reason
        
        # Measure Answer Relevancy
        relevancy_metric.measure(test_case)
        r_score = relevancy_metric.score
        r_reason = relevancy_metric.reason
        
        status = "SUCCESS"
    except Exception as e:
        f_score, f_reason = 0.0, f"Error: {e}"
        r_score, r_reason = 0.0, f"Error: {e}"
        status = f"FAILED: {e}"

    # Save live after every question
    eval_results[qid] = {
        "question_id": qid,
        "question": item["question"],
        "rag_answer": item["rag_answer"],
        "faithfulness_score": f_score,
        "faithfulness_reason": f_reason,
        "relevancy_score": r_score,
        "relevancy_reason": r_reason,
        "status": status,
        "eval_time_sec": time.time() - t_case_start
    }
    
    with open(EVAL_OUT_FILE, "w") as f:
        json.dump(eval_results, f, indent=2)
        
    if idx % 10 == 0 or idx == len(to_evaluate):
        scores = [v for v in eval_results.values() if v["status"] == "SUCCESS"]
        avg_f = sum(s["faithfulness_score"] for s in scores) / len(scores) if scores else 0
        avg_r = sum(s["relevancy_score"] for s in scores) / len(scores) if scores else 0
        
        elapsed = time.time() - t_start
        est_remaining = (elapsed / idx) * (len(to_evaluate) - idx)
        
        print(f"Eval [{idx}/{len(to_evaluate)}] | Running Avg: Faithfulness={avg_f:.2%}, Relevancy={avg_r:.2%} | Est. remaining: {est_remaining/60:.1f} mins", flush=True)

# ── 6. Print Final Summary Table & Save to File ──────────────────────────────
scores = [v for v in eval_results.values() if v["status"] == "SUCCESS"]
avg_f = sum(s["faithfulness_score"] for s in scores) / len(scores) if scores else 0
avg_r = sum(s["relevancy_score"] for s in scores) / len(scores) if scores else 0
overall = (avg_f + avg_r) / 2

report_lines = [
    "# 🏆 DeepEval Local RAG Evaluation Report",
    "",
    "## 📊 Summary Performance Metrics",
    "",
    "| Metric | Score | Description |",
    "| :--- | :--- | :--- |",
    f"| **Faithfulness** | **{avg_f:.2%}** | Factual consistency (no hallucinations) |",
    f"| **Answer Relevancy** | **{avg_r:.2%}** | How well the RAG answer fits the question |",
    f"| **OVERALL RAG SCORE** | **{overall:.2%}** | Combined performance average |",
    "",
    f"**Total Questions Evaluated:** {len(scores)}",
    "",
    "---",
    "## 📝 Sample Evaluation Results (First 5)",
    "",
    "| Question ID | Question | Faithfulness | Relevancy | Status |",
    "| :--- | :--- | :---: | :---: | :---: |"
]

for s in list(scores)[:5]:
    q_truncated = s["question"][:60] + "..." if len(s["question"]) > 60 else s["question"]
    report_lines.append(
        f"| {s['question_id']} | {q_truncated} | {s['faithfulness_score']:.2%} | {s['relevancy_score']:.2%} | {s['status']} |"
    )

report_content = "\n".join(report_lines)

print("\n" + "="*55)
print("🏆   DEEPEVAL RAG EVALUATION REPORT SUMMARY")
print("="*55)
print(f"  Total Evaluated Questions : {len(scores)}")
print(f"  Average Faithfulness      : {avg_f:.2%} (Factual consistency)")
print(f"  Average Answer Relevancy  : {avg_r:.2%} (Answers the question)")
print(f"  Overall RAG Score         : {overall:.2%}")
print("="*55)

with open(REPORT_FILE, "w") as f:
    f.write(report_content)
print(f"\n📄 Saved report table to: {REPORT_FILE}")
