# 🏆 DeepEval Hybrid RAG Evaluation Report (500 Questions)

This is the final evaluation report generated on the first **500 questions** of the dataset using the **Hybrid Search (Vector + BM25 Keyword Search + Cross-Encoder Reranker)** strategy.

## 📊 Summary Performance Metrics

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Faithfulness** | **31.10%** | Factual consistency (no hallucinations) |
| **Answer Relevancy** | **58.30%** | How well the RAG answer directly fits the question |
| **OVERALL RAG SCORE** | **44.70%** | Combined performance average |

* **Total Questions Evaluated:** 500 / 500 (100% complete)
* **Evaluator Model:** Qwen 2.5 3B Instruct (Local GPU)
* **Database Strategy:** Fast Ingestion (Text-only, No OCR)
* **Retrieval Strategy:** Dense Vector (BGE Large) + Sparse Keyword (BM25) fused with RRF (Reciprocal Rank Fusion) and reranked using Cross-Encoder.

---

## 📈 Analysis & Insights

1. **Major Relevancy Boost (+23.3% Improvement)**:
   * Relevancy jumped from **35.00%** (Baseline Vector Search) to **58.30%** (Hybrid Search).
   * This proves that adding **BM25 keyword search** helps locate specific acronyms, variables, figures, and technical terms in scientific papers that dense vector search matches poorly on.

2. **Faithfulness Bottleneck (31.10%)**:
   * While Faithfulness improved slightly (from ~27% to 31.10%), it remains low because the database was created using **fast text-only mode (No OCR)**. 
   * Many questions ask about values in tables or variables inside mathematical formulas. Since these were skipped or scrambled during plain-text extraction, the model either hallucinated or guessed. 
   * **Solution**: Implementing the **Hi-Res strategy** (which parses tables and OCRs figures) will be required to resolve this bottleneck.

---

## 📝 Sample Evaluation Results (First 5)

| Question ID | Question | Faithfulness | Relevancy | Status |
| :--- | :--- | :---: | :---: | :---: |
| **1407.7906_Q01** | What is the primary purpose of target propagation as proposed in this paper, and how does it relate to back-propagation? | 50.00% | 50.00% | SUCCESS |
| **1407.7906_Q02** | In Figure 1, what three main actors are illustrated for an intermediate layer h in the generative model? | 0.00% | 0.00% | SUCCESS |
| **1407.7906_Q03** | How is the KL-divergence between Q and P decomposed in Equation (3)... | 0.00% | 25.00% | SUCCESS |
| **1407.7906_Q04** | What is difference target propagation (DTP) and why is it introduced... | 50.00% | 75.00% | SUCCESS |
| **1407.7906_Q05** | How are the targets for the hidden layers computed in difference target propagation... | 75.00% | 75.00% | SUCCESS |
