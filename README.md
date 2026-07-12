# 📚 Multimodal Multi-PDF Research Assistant (RAG)

A professional, modular Research Paper analysis assistant driven by **Retrieval-Augmented Generation (RAG)**. This system supports layout-aware PDF ingestion, dual-store FAISS indexing (Text + Diagrams/Charts), cross-encoder reranking, and an intelligent LLM-based query intent router.

---

## 🛠️ System Architecture

The following diagram illustrates the hybrid RAG retrieval pipeline and visual processing flow:

```mermaid
graph TD
    PDF[Upload PDF File] --> Parser{PDF Parser Strategy}
    Parser -->|Fast Strategy| TextOnly[Extract Text Chunks]
    Parser -->|Hi-Res Strategy| LayoutParser[Layout Parsing]
    
    LayoutParser --> TextChunks[Extract Paragraph Elements]
    LayoutParser --> ImageCrop[Crop Diagram / Table Figures]
    
    TextOnly --> EmbedText[BGE Text Embedder]
    TextChunks --> EmbedText
    ImageCrop --> EmbedImage[CLIP Image Embedder]
    
    EmbedText --> FAISS_Text[(FAISS Text Index)]
    EmbedImage --> FAISS_Image[(FAISS Visual Index)]
```

---

## 🤖 LLM Intent Router & Context Filtering

Instead of hardcoded keyword checks, queries are routed through a dynamic **LLM Intent Router** powered by Gemini. This allows resolving natural language file references ("summarize the last two papers") and applying precise target document filters during search.

```mermaid
graph TD
    UserQuery["User Input:<br>'Compare the attention paper and the social engineering document'"] 
    --> Router["LLM Intent Router<br>(Gemini JSON Call)"]
    
    Router --> JSON["Structured JSON Output"]
    
    JSON --> |"intent: COMPARISON"| Compare["Multi-Paper Comparison<br>(Generates Side-by-Side Table)"]
    JSON --> |"intent: SECTION_SUMMARY"| Section["Section-Wise Summary<br>(Groups chunks by section headers)"]
    JSON --> |"intent: SUMMARY"| Summary["Executive Summary<br>(Consolidated Document Summary)"]
    JSON --> |"intent: STANDARD_CHAT"| QnA["Standard Contextual Q&A<br>(Filters RAG database to target files)"]
    
    style UserQuery fill:#1e1e2e,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4
    style Router fill:#1e1e2e,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4
    style JSON fill:#181825,stroke:#a6e3a1,stroke-width:2px,color:#cdd6f4
    style Compare fill:#11111b,stroke:#f38ba8,stroke-width:1px,color:#a6adc8
    style Section fill:#11111b,stroke:#f38ba8,stroke-width:1px,color:#a6adc8
    style Summary fill:#11111b,stroke:#f38ba8,stroke-width:1px,color:#a6adc8
    style QnA fill:#11111b,stroke:#f38ba8,stroke-width:1px,color:#a6adc8
```

---

## 🛡️ High-Availability LLM Gate Architecture

To ensure zero downtime when running under restricted free-tier API quotas, all LLM requests are managed by a **Unified LLM Gate Handler**. If the primary gate fails (due to 429 quota exhaustion or rate limits), the handler dynamically redirects subsequent queries to the secondary gate without user intervention.

```mermaid
graph TD
    subgraph Client Application Layer
        AppCore["Router / Answering / Summarizer / Comparison"]
    end

    subgraph LLM Gate Layer
        AppCore -->|Unified generate call| GateHandler["LLM Gate Handler"]
        
        GateHandler -->|Gate 1| GeminiGate["Gemini Provider"]
        GeminiGate -->|429 / Quota Failure| GateHandler
        
        GateHandler -->|Gate 2: Auto Failover| ORGate["OpenRouter Provider"]
    end
    
    style AppCore fill:#1e1e2e,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4
    style GateHandler fill:#181825,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4
    style GeminiGate fill:#11111b,stroke:#f38ba8,stroke-width:1px,color:#a6adc8
    style ORGate fill:#11111b,stroke:#a6e3a1,stroke-width:1px,color:#a6adc8
```

You can view the full design diagram at [docs/llm_gate_architecture_diagram.png](file:///Users/nanibayanaboina2750/Desktop/research-rag-assistant/docs/llm_gate_architecture_diagram.png).

---

## 📊 Evaluation Matrix & Performance Results

We executed a comprehensive RAG quality audit benchmarking **220 total questions** across three distinct configurations:

### 1. 2x2 Local RAG Matrix (Graded via Qwen 2.5 3B local judge)
*Tested across 200 benchmark questions sharded in parallel:*

| Retrieval Strategy | Document Source (Fast vs Hi-Res) | Faithfulness | Answer Relevancy | Overall RAG Score | Progress |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline RAG** <br>*(Dense vector search)* | **Fast (Text-Only)** | 27.13% | 35.00% | **31.06%** | *Baseline* |
| **Baseline RAG** <br>*(Dense vector search)* | **Hi-Res (OCR/Layout)** | 30.21% | 54.61% | **42.41%** | **+11.35%** 📈 |
| **Hybrid RAG** <br>*(Dense + Sparse BM25)* | **Fast (Text-Only)** | 31.10% | 58.30% | **44.70%** | **+13.64%** 📈 |
| **Hybrid RAG** <br>*(Dense + Sparse BM25)* | **Hi-Res (OCR/Layout)** | **39.45%** | **57.68%** | **48.57%** | **+17.51%** 🏆 |

### 2. Production Multimodal Gemini RAG Matrix (Graded via Llama 3.1 8B)
*Evaluates Gemini 2.5 Flash on 20 pilot questions incorporating text + visual CLIP-retrieved diagrams:*

*   **Faithfulness:** **`88.75%`** (highly grounded, near-zero hallucinations)
*   **Answer Relevancy:** **`83.75%`**
*   **OVERALL RAG SCORE:** **`86.25%`** 🏆

### 3. Multimodal Features Quality Audit (Graded via Llama 3.3 70B Free)
*Evaluates our specialized layout-aware summarization and side-by-side paper comparisons:*

*   **Seq2Seq Summary Quality:** **`100.00%`** 🏆 (Successfully extracts exact training metrics like 34.81 BLEU score and maps to attached LSTM diagrams).
*   **Target Propagation Summary Quality:** **`100.00%`** 🏆 (Successfully extracts and matches VAE loss and KL-divergence formulas).
*   **Side-by-Side Paper Comparison Quality:** **`100.00%`** 🏆 (Flawlessly contrasts methodologies, architectures, and limitations).

---

## ✨ Key Features

*   **Multimodal RAG Retrieval:** Fetches both high-relevance paragraphs and visual diagrams/charts, attaching figures natively to the Gemini generation payload.
*   **High-Availability OpenRouter Fallback:** Automatically switches to OpenRouter API (converting visual diagrams to base64 JPEG data URIs) if the primary Google Gemini keys hit rate limits or quota boundaries.
*   **Non-Blocking Indexing:** Runs PDF layout parsing and indexing inside background worker threads, keeping the Streamlit UI completely responsive.
*   **Hybrid Summarization Router:** Calculates paper character lengths and dynamically chooses between Direct Single-Pass (fast, detailed) and Hierarchical Map-Reduce (section summaries -> merge) to optimize token costs.
*   **CLIP Threshold Filtering:** Filters out unrelated visual results using a cosine similarity threshold of `0.28`.
*   **Cross-Encoder Reranking:** Re-evaluates chunk relevance using a MiniLM Cross-Encoder to optimize RAG accuracy.

---

## 📂 Project Structure

```
research-rag-assistant/
├── config.py                 # Central configurations (models, paths, index dimensions)
├── main.py                   # Master CLI subcommand entry point
├── app/                      # Web dashboard application
│   ├── components/           # Modular UI widgets
│   │   ├── chat.py           # Chat bubble rendering & RAG routing execution
│   │   ├── sidebar.py        # File uploads, strategy select, background threads
│   │   └── router.py         # LLM-based intent routing & reference resolution
│   └── ui.py                 # Streamlit entry point
├── src/                      # RAG Package Source Code
│   ├── core/
│   │   ├── generator/        # Gemini client wrapper, Q&A, and summarization
│   │   ├── ingestion.py      # PDF parsing and document layouts
│   │   ├── chunker.py        # Element chunking & section tagging
│   │   ├── embedder.py       # Local BGE and CLIP embedding generation
│   │   └── vector_store.py   # FAISS dual index database management
├── evaluation_suite/         # Complete RAG Evaluation & Benchmarking Suite
│   ├── EVALUATION_GUIDE.md   # Step-by-step benchmark guide
│   ├── dataset/              # Gold standard benchmark datasets (200 QA)
│   ├── evaluations/          # Qwen/Gemini matrix markdown reports & JSONs
│   ├── pipelines/            # Evaluation runners and LLM judges
│   ├── scripts/              # Ingestion utilities and downloader scripts
│   └── logs/                 # Sharded run logs and retrieved answers
├── archives/                 # Backup zip archives and testing folders
├── development_phases/       # Historic developmental phase code (Phases 1-6)
```

---

## 📈 Running Evaluations

All benchmarking datasets, LLM judges, sharded run logs, and detailed reports are managed inside the `evaluation_suite/` directory.

To run the evaluations and read step-by-step guidelines, refer to the **[Evaluation Guide](file:///Users/nanibayanaboina2750/Desktop/research-rag-assistant/evaluation_suite/EVALUATION_GUIDE.md)**:
👉 **[`evaluation_suite/EVALUATION_GUIDE.md`](file:///Users/nanibayanaboina2750/Desktop/research-rag-assistant/evaluation_suite/EVALUATION_GUIDE.md)**

