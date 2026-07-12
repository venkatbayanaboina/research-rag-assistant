# 📚 Research RAG Assistant: Complete Evaluation Guide

This guide details the methodology, metrics, and procedures used to test and evaluate the Research RAG Assistant across various architectures and models.

---

## 📈 1. Core Metrics: The RAG Triad
We use the industry-standard **DeepEval** framework to evaluate generated answers based on two primary dimensions:

### A. Faithfulness (Groundedness)
* **What it measures:** Factual consistency and the presence of hallucinations.
* **Calculation:** The judge model extracts claims from the RAG Answer and verifies whether each claim is explicitly supported by the retrieved Context.
* **Score range:** `0%` (completely hallucinated) to `100%` (completely grounded).

### B. Answer Relevancy
* **What it measures:** How directly and cleanly the RAG Answer addresses the user query.
* **Calculation:** The judge checks for conversational fluff, off-topic statements, or incomplete answers.
* **Score range:** `0%` (off-topic) to `100%` (perfectly relevant).

---

## 📊 2. Evaluation Phase 1: Local Qwen 2x2 Matrix
* **Target Count:** 200 Questions (sharded across 5 parallel runs).
* **Generator Model:** Local Qwen 2.5 3B (loaded in 16-bit precision on GPU).
* **Grader Model:** Local Qwen 2.5 3B.
* **Tested Configurations:**

| Retrieval Strategy | Document Source (Fast vs Hi-Res) | Description |
| :--- | :--- | :--- |
| **Baseline RAG** | **Fast (Text-Only)** | Dense vector search (BGE-Large) on plain text. |
| **Baseline RAG** | **Hi-Res (OCR/Layout)** | Dense vector search on layout-parsed PDFs (includes table text). |
| **Hybrid RAG** | **Fast (Text-Only)** | Dense vector (BGE) + Sparse Keyword (BM25) fused via RRF. |
| **Hybrid RAG** | **Hi-Res (OCR/Layout)** | Dense + Sparse RRF search on layout-parsed database + Cross-Encoder reranking. |

---

## 🚀 3. Evaluation Phase 2: Production Multimodal Gemini RAG
* **Target Count:** 20 Questions pilot.
* **Generator Model:** `gemini-2.5-flash` / `gemini-2.0-flash` (via OpenRouter to bypass direct quota limits).
* **Grader Model:** `meta-llama/llama-3.1-8b-instruct` (via OpenRouter).
* **Key Enhancements:**
  1. **Visual Context:** Queries the CLIP visual index to find the best matching figure/table and sends the actual PIL Image to Gemini.
  2. **Expanded Context:** Feeds the top 5 reranked context blocks (~2 pages of text).
  3. **Multimodal Reasoning:** Gemini synthesizes both text and visual pixels to produce highly grounded results.

---

## 🎯 4. Evaluation Phase 3: Multimodal Feature Audit
This phase validates specialized scientific features: **summarization** and **side-by-side paper comparison**.
* **Generator Model:** `nvidia/nemotron-nano-12b-v2-vl:free` (100% free multimodal model).
* **Grader Model:** `openrouter/free` (automatically routes to active free judges, like Llama 3 8B or Mistral 7B).

### A. Multimodal Summarization
* **Hybrid Summarization Router:** Calculates paper sizes. If the text is under 60k characters, it runs a **Direct Single-Pass**. If over 60k, it runs a **Hierarchical Map-Reduce** (section summaries -> merge).
* **Visual Attachment:** Automatically attaches the model's architecture diagram, performance table, and main formulas (KL-divergence, loss functions) as PIL images to the query.

### B. Multimodal Side-by-Side Paper Comparison
* Joins text for both papers.
* Extracts and attaches key visuals (diagrams & tables) for both papers.
* Prompts the generator to construct a detailed comparative Markdown table contrasting architectures, formulas, results, and limitations.

---

## 📂 5. Evaluation Directory File Structure

The evaluations directory is structured as follows:
```text
evaluation_suite/
├── dataset/                           # Dataset & QA generator scripts
│   ├── gold_qa_dataset.json           # 200 question benchmark dataset
│   └── generate_eval_dataset.py       # QA parser script
├── pipelines/                         # Generation & RAG pipeline scripts
│   ├── rag_hybrid_generator.py        # Core RAG loop
│   └── deepeval_local_judge.py        # Local Qwen evaluator logic
├── evaluations/                       # Evaluation reports & scores
│   ├── local_qwen/                    # Local 2x2 matrix MD reports & raw JSONs
│   ├── production_gemini/             # Gemini RAG MD reports & raw JSONs
│   └── features_audit/                # Multimodal summary & comparison folders
├── scripts/                           # File utilities & batch downloader tools
├── logs/                              # Shard logs & retrieved JSON answers
└── configs/                           # Raw PDF files and download URLs
```

---

## 💻 6. How to Run Evaluations

### A. Run Local Qwen Evaluation (200 Questions)
Ensure your local Ollama server is running with `ollama run qwen2.5:3b`, then execute:
```bash
python pipelines/run_rag_evaluation.py
```

### B. Run Multimodal Summarization & Comparisons (in Colab)
Paste the Colab scripts in your notebook and run them to generate and download the packaged zip containing:
* `summaries/` (Detailed markdown summaries referencing attached equations/figures).
* `comparisons/` (Side-by-side markdown comparison tables).
* `deepeval_multimodal_features_report.md` (The OpenRouter judge report).
