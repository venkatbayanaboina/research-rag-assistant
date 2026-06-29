# RAG Production Evaluation & Confidence Testing

Standard software logging (like database execution times or API success rates) cannot tell you if an LLM is hallucinating or generating irrelevant answers. To solve this, production-grade AI systems deploy automated evaluation pipelines.

This document describes the testing architecture set up for the Multimodal Dual-Store RAG Assistant using **DeepEval** ("Pytest for LLMs") and **Ragas** metrics.

---

## 📐 The RAG Triad of Metrics

Our test cases measure two primary metrics to score generation quality (on a scale of `0.0` to `1.0`):

```
                       [ User Query ]
                             |
                   +---------+---------+
                   |                   |
         (Answer Relevancy)      (Context Precision)
                   |                   |
                   v                   v
           [ Generated Answer ] <------------ [ Retrieved Context ]
                                (Faithfulness)
```

### 1. Faithfulness (Groundedness)
*   **What it measures:** Is the generated answer derived *only* from the retrieved document chunks?
*   **Why it matters:** Prevents **hallucinations**. If the model introduces external facts not present in your indexed PDFs (even if true in the real world), the Faithfulness score drops.
*   **Calculation:** Uses LLM-as-a-judge to segment the output into individual claims and verify if each claim is supported by the retrieval context.

### 2. Answer Relevancy
*   **What it measures:** Does the generated answer directly address the user's question?
*   **Why it matters:** Prevents the model from generating detailed, well-grounded paragraphs that are completely off-topic or fail to answer the specific query.
*   **Calculation:** Evaluates the semantic match between the input prompt and the output answer.

---

## 🛠️ Testing Infrastructure: DeepEval + Pytest

We use **DeepEval** because it is built to integrate directly with Python's standard `pytest` testing ecosystem. This lets us run quality checks locally or trigger them as standard unit tests during GitHub Actions CI/CD runs.

### Test Architecture
*   **Core Execution:** The test runner executes `execute_rag_pipeline(prompt, indexed_docs)` to fetch actual search contexts and model answers.
*   **Evaluation Case:** Constructs a `LLMTestCase` containing:
    *   `input`: The user query.
    *   `actual_output`: The model's answer.
    *   `retrieval_context`: The text list of FAISS database hits.
*   **Quality Judge:** Uses the active LLM gate (Gemini or OpenRouter fallback) as the algorithmic judge to score the test case.
*   **Pass/Fail Threshold:** Set to `0.7` by default. If any code changes or database modifications drop the score below `0.7`, the test suite fails.

---

## 🚀 Execution Guide

### 1. Install Evaluation Dependencies
Ensure `deepeval` and `pytest` are installed in your virtual environment:
```bash
pip install deepeval pytest
```

### 2. Run the Quality Tests
Run the automated evaluation suite from the terminal:
```bash
pytest -s tests/test_rag_eval.py
```

### 3. Reviewing DeepEval Dashboard (Optional)
DeepEval provides a local web dashboard to trace historical quality scores across git commits:
```bash
deepeval login
```

---

## ⚡ Scaling up with Local LLM Evaluation in Colab (Ollama)

For large golden datasets (e.g. 50+ test cases), running LLM-as-a-judge evaluations via Cloud APIs is highly restricted by free-tier rate limits (15 Requests Per Minute on Gemini, and 100 Requests Per Day on OpenRouter).

To evaluate 50+ test cases **completely for free and with zero rate limits**, you can spin up **Ollama** with **Llama-3** directly inside your Google Colab notebook. The test suite automatically detects the local Ollama server and routes all judge calls to it.

### Setup Instructions for Google Colab

1. **Install Ollama in Colab:**
   ```bash
   !curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Start the Ollama Server in the background:**
   ```python
   import subprocess
   import time

   # Launch the Ollama background daemon
   subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
   time.sleep(5) # Allow 5 seconds to initialize
   ```

3. **Download Llama-3 (8B):**
   ```bash
   !ollama pull llama3
   ```

Once Llama-3 is pulled, run `!pytest tests/test_rag_eval.py`. The test suite will log `SYSTEM LOG: Local Ollama server detected. Using local Llama3 as the evaluation judge.` and execute the evaluations locally using Colab's GPU.
