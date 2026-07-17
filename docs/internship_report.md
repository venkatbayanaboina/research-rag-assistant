# 🎓 Internship Project Report: Multimodal Scientific Research RAG Assistant

---

### **Metadata**
* **Project Title:** Production-Grade Multimodal Scientific Research RAG Assistant
* **Internship Role:** AI/RAG Engineering Intern
* **Development Domain:** Large Language Models (LLMs), Multimodal Databases, Hardware-Accelerated Vector Stores, and High-Performance Web UI Pipelines
* **Key Technologies:** Python, PyTorch (Metal Performance Shaders - MPS), FAISS, Hugging Face `transformers` (BGE-Large, CLIP, Cross-Encoder), Streamlit, Gemini API, DeepEval, Playwright
* **Repository Link:** [research-rag-assistant](https://github.com/venkatbayanaboina/research-rag-assistant)

---

## 📋 1. Executive Summary
This report summarizes the design, implementation, optimization, and evaluation of a production-grade **Multimodal Scientific Research Retrieval-Augmented Generation (RAG) Assistant** developed during the internship. 

Academic and scientific research papers are heavily reliant on multi-dimensional structures: text narratives, mathematical formulas, structural charts, and data tables. Traditional text-only RAG pipelines fail to retrieve or reason about visual figures and complex layouts, resulting in factual omissions and hallucinations. 

To address this, this project delivers a production-grade system that parses PDF structures, indexes text in FAISS via BGE embeddings, indexes visual layout components using CLIP embeddings, and dynamically orchestrates query-routing to deliver factual Q&A, Map-Reduce summarizations, and multi-document comparative matrices (up to 5+ papers simultaneously) evaluated with standard RAG metrics.

---

## 🛠️ 2. System Architecture & Design
The application is decoupled into five distinct pipeline layers, ensuring modularity, easy maintenance, and scalability.

```
[ PDF Document ] ──> [ parsing & extraction (unstructured / pypdf) ]
                           │
                           ├──> Text Chunks ──> BGE Embedding ──> FAISS Text DB (1024-dim)
                           │
                           └──> Visual Cropped Figures ──> CLIP ──> FAISS Image DB (512-dim)
                                                                       │
[ User Prompt ] ──> [ Intent Router ] ──> [ Context Retrieval & Reranker ] ──> [ Gemini LLM ] ──> [ Answer ]
```

### A. Document Ingestion & Parsing Strategy
The ingestion engine support two distinct processing pathways:
1. **`fast` Strategy (PyPDF):** Strips raw text in milliseconds; suitable for text-heavy pre-parsed documents.
2. **`hi_res` Strategy (Unstructured + YOLOX):** Employs a local deep-learning model to detect document layout structures. It identifies headers, text bodies, tables, and charts, cropping tables and charts as standalone images.

### B. Semantic Chunking (Chunk-by-Title)
Instead of arbitrary character-count boundaries, a custom chunking algorithm aggregates text under section headers. This semantic boundary preservation prevents the loss of context when formulas or descriptions span across paragraphs.

### C. Hardware-Accelerated Vector Store (FAISS)
Generates high-dimensional embeddings for local GPU/MPS execution:
* **Text Index:** Generates 1024-dimensional embeddings using `BAAI/bge-large-en-v1.5` on Apple Silicon MPS, stored in a FAISS FlatIP index.
* **Image Index:** Generates 512-dimensional embeddings using CLIP `clip-ViT-B-32`, mapping cropped figure coordinates to vector indices.

### D. Intent Routing & Orchestration
Every incoming user prompt is classified before execution. Rather than running standard vector lookups on all queries, a router classifies intent into:
* **`STANDARD_CHAT`**: Basic RAG query, fetches top text chunks and CLIP images, reranks, and answers.
* **`SUMMARY`**: Map-Reduce summarization for large papers to prevent context window bloat.
* **`SECTION_SUMMARY`**: Executive section-by-section breakdown of a paper's chapters.
* **`COMPARISON`**: Assembles comparative matrices dynamically for up to 5+ target documents.

---

## 🚀 3. Key Technical Challenges Solved (Engineering Highlights)
During development, several critical concurrency, resource, and thread-safety issues were identified and resolved to make the application production-grade:

### ⚠️ Challenge 1: PyTorch CPU Segmentation Faults on macOS (Apple Silicon)
* **Problem:** Under load, running PyTorch CPU thread-pools alongside Hugging Face `tokenizers`' native Rust multiprocessing caused a hard segmentation fault (`[1] segmentation fault python3 main.py query`) and leaked semaphore locks.
* **Solution:** 
  1. Set `os.environ["TOKENIZERS_PARALLELISM"] = "false"` to prevent concurrent tokenizer forks.
  2. Implemented `torch.set_num_threads(1)` to limit PyTorch CPU thread contention.
  3. Routed both text and visual model loading directly onto **`mps`** (Metal Performance Shaders GPU), bypassing CPU multi-threading bottlenecks and accelerating embedding times.

### ⚠️ Challenge 2: Streamlit Infinite Rerun Loops & 99% CPU Lockup
* **Problem:** During background ingestion, the UI progress checking code called `st.rerun()`. Because the clearing statement was located *after* the rerun check, the script was interrupted and entered an infinite recursive loop, locking 99.0% of a CPU core.
* **Solution:** Implemented a **Safe State Transition Pattern** where the checking fragment copies the success/error messages to `st.session_state` and **wipes the global variables immediately** *before* calling `st.rerun()`, ensuring the next execution cycle renders cleanly.

### ⚠️ Challenge 3: Ingestion Concurrency and Lock Deadlocks
* **Problem:** A non-reentrant database lock (`db_lock = threading.Lock()`) caused threads to deadlock when an index update failed or when nested calls tried to read registries.
* **Solution:** Audited and restructured the database modules to isolate lock acquisition strictly to the immediate read/write boundaries, wrapping updates in `try...finally` safety blocks to guarantee lock release in any failure scenario.

### ⚠️ Challenge 4: Multi-Paper Comparison Scale Limitations
* **Problem:** The comparative RAG engine was hardcoded to evaluate exactly two papers (`doc_a` and `doc_b`). Attempting to compare more papers ignored the remaining targets.
* **Solution:** Redesigned the Markdown table compiler and prompt constructor to support **dynamic column allocation**, accepting lists of 2 to 5+ papers and building a structured side-by-side comparative table containing equal depth for all selected targets.

---

## 📊 4. Evaluation Suite & Benchmark Results
To evaluate the quality and correctness of the RAG pipeline, a benchmark evaluation was executed:

### A. RAG Triad Metrics
We integrated the **DeepEval** framework to measure three core RAG dimensions:
1. **Faithfulness (Groundedness):** Verifies that LLM claims are backed by the retrieved text context.
2. **Answer Relevancy:** Measures how cleanly the answer addresses the prompt.
3. **Correctness (vs Gold Standard):** Uses an LLM judge to evaluate RAG responses against a golden benchmark dataset.

### B. Multimodal Feature Audit (5-Paper Evaluation Results)
We ran the evaluation suite against 5 target papers currently indexed in the system, generating individual summaries and comparison matrices, and grading them using Gemini as the LLM-as-Judge.

The generated report **[`deepeval_multimodal_features_report.md`](file:///Users/nanibayanaboina2750/Desktop/research-rag-assistant/evaluation_suite/evaluations/features_audit/deepeval_multimodal_features_report.md)** details the following results:

| Feature / Output File | Evaluated Metric / Score | Reasoning / Judge's Critique |
| :--- | :---: | :--- |
| **`Summary_attention-is-all-you-need-Paper_summary.md`** | **98.50%** | Comprehensive coverage of encoder/decoder stacks, parameters, BLEU scores, and critical equations. |
| **`Summary_1408.5882_Sentence_Classification_summary.md`** | **98.50%** | Accurately details CNN layers, static/non-static word vectors, and dropout math. |
| **`Summary_1407.7906_Target_Propagation_summary.md`** | **98.00%** | Deeply captures credit assignment mathematical frameworks and propagation details. |
| **`Summary_1409.3215_Sequence_to_Sequence_summary.md`** | **98.50%** | Factually reports BLEU scores (34.81) and the sentence-reversal methodology. |
| **`Summary_Social-Engineering-Attack_summary.md`** | **98.00%** | Evaluates attack topologies and human vulnerability statistics accurately. |
| **`Comparison_5_Papers_Comparative_Analysis_Matrix.md`** | **98.50%** | Generates a structured 6-column matrix contrasting methodologies, formulas, and limitations. |
| **OVERALL SYSTEM FEATURE SCORE** | **98.33%** | **Highly grounded, clear, structured, and free of hallucinations.** |

---

## 📈 5. Key Learnings & Professional Growth
During the course of the internship, significant expertise was gained in:
1. **Multimodal Data Modeling:** Representing non-text elements (images, formulas, layout geometries) as semantic vectors using CLIP.
2. **Hardware Acceleration:** Compiling and running model weights (BGE, CLIP) on macOS GPU structures via PyTorch Metal Performance Shaders (`mps`).
3. **Thread-Safe Architecture:** Writing robust, concurrent web server pipelines in Streamlit using fragments, session variables, and mutex locks.
4. **LLM-as-Judge Frameworks:** Structuring evaluation pipelines with DeepEval to benchmark RAG responses quantitatively.

---

## 🏁 6. Conclusion
The internship successfully delivered a production-grade, hardware-accelerated **Multimodal RAG Assistant**. By addressing layout extraction, visual indexing, reranking, and dynamic query routing, the system provides a highly accurate interface for scientific research. With an overall feature score of **98.33%** and key macOS hardware optimizations, this project represents a significant advancement over standard text-only RAG pipelines.
