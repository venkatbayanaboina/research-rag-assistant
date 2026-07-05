# 📚 RAG Pipeline Evaluation Framework

This directory contains the scripts, schemas, and benchmark datasets used to generate QA evaluation pairs and test the accuracy of the Retrieval-Augmented Generation (RAG) assistant.

---

## 📅 Roadmap & Execution Flow

The evaluation framework consists of **5 sequential phases** designed to isolate database indexing, question generation, and LLM-as-a-judge scoring.

```
  Phase 1: Ingestion          Phase 2: QA Generation      Phase 3: RAG Retrieval      Phase 4: LLM Answering      Phase 5: DeepEval Judge
    (Colab GPU)                   (Local Chrome)                 (Local)                     (Local)                      (Local)
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│  Build FAISS Index   │ ──>│  Generate Gold QAs   │ ──>│  Retrieve DB Chunks  │ ──>│  Generate RAG Answer │ ──>│ Score RAG vs ChatGPT │
│                      │    │                      │    │                      │    │                      │    │                      │
│ Output:              │    │ Output:              │    │ Output:              │    │ Output:              │    │ Output:              │
│ no_ocr_and_fast_mode │    │ gold_qa_dataset      │    │ rag_retrieved_       │    │ rag_retrieved_       │    │ deep_eval_results    │
│ .zip                 │    │ .json                │    │ answers.json         │    │ answers.json         │    │ .json                │
└──────────────────────┘    └──────────────────────┘    └──────────────────────┘    └──────────────────────┘    └──────────────────────┘
```

---

## 🚶 Step-by-Step Instructions

### 📂 Phase 1: PDF Ingestion (Google Colab GPU)
* **Goal**: Build the vector database index on a Google Colab T4 GPU to parse text and crop figures/tables without OCR.
* **Why Colab?**: Running the YOLOX layout parsing and BGE-Large/CLIP embedding models locally on CPU for 315 papers takes hours. A Colab T4 GPU finishes the entire ingestion in **~15 minutes**.
* **OCR Optimization**: To maximize ingestion speed, the pipeline completely bypasses OCR (`ocr_languages="eng"`), relying on the PDF's native selectable digital text. Additionally, a `ThreadPoolExecutor` with **3 parallel threads** processes 3 papers concurrently.
* **Execution**:
  1. Zip your local PDFs: `zip -r papers_pdfs.zip "papers download/pdfs"`
  2. Open Colab and connect to a **T4 GPU** runtime.
  3. Upload `papers_pdfs.zip` to Colab.
  4. Copy and run the setup and ingestion cells from [colab_ingestion_guide.md](file:///Users/nanibayanaboina2750/.gemini/antigravity/brain/ade3cade-6db7-41f6-9e90-1d91ffbcd0f9/colab_ingestion_guide.md).
  5. Zip the resulting database: `!zip -r no_ocr_and_fast_mode.zip storage`
  6. Download the zip and extract its contents directly into your local `research-rag-assistant/storage/` folder.

---

### 📝 Phase 2: Gold QA Dataset Generation (API & Sharding)
* **Goal**: Generate a structured benchmark dataset containing **15 research-grade QA pairs per academic paper**.
* **Direct API Script (⚡ Highly Recommended)**: [`generate_dataset_api.py`](file:///Users/nanibayanaboina2750/Desktop/research-rag-assistant/papers download/generate_dataset_api.py)
  * Directly queries Google AI Studio using your `GEMINI_API_KEY`. 
  * Bypasses the browser completely. Takes ~3 seconds per paper.
  * Respects the **15 RPM free tier limit** to remain 100% free (takes ~21 minutes for 315 papers).
* **Parallel API Sharding (⚡ Super Fast & Free)**:
  If you have **two free Gemini API keys** (from different Google accounts), you can run two processes in parallel. This partitions the work in half, completing all 315 papers in **10.5 minutes for free**:
  ```bash
  # In terminal 1 (Process 1 handles shard 1):
  GEMINI_API_KEY="api_key_one" python3 generate_dataset_api.py --shard 1 --num-shards 2

  # In terminal 2 (Process 2 handles shard 2):
  GEMINI_API_KEY="api_key_two" python3 generate_dataset_api.py --shard 2 --num-shards 2
  ```
* **Primary Web Script (Browser Fallback)**: [`automate_gemini.py`](file:///Users/nanibayanaboina2750/Desktop/research-rag-assistant/papers download/automate_gemini.py) (uses Gemini Web App).
* **Secondary Web Script (Browser Fallback)**: [`automate_chatgpt.py`](file:///Users/nanibayanaboina2750/Desktop/research-rag-assistant/papers download/automate_chatgpt.py) (uses ChatGPT Web App).

---

#### 🛠️ Automation & Connection Mechanism
The script bypasses API costs and rate limits by running browser automation directly on your locally authenticated Google Chrome browser:
1. **Remote Debugging Mode**: Launch Chrome with Chrome DevTools Protocol (CDP) enabled on port `9222`:
   ```bash
   /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 --user-data-dir="/Users/nanibayanaboina2750/ChromeProfile"
   ```
2. **Playwright Connection**: Instead of starting a headless/blank browser, Playwright attaches directly to this active debugging session:
   ```python
   browser = p.chromium.connect_over_cdp("http://localhost:9222")
   default_context = browser.contexts[0]
   page = default_context.pages[0]
   ```
   This retains your active login session cookies for `gemini.google.com` (or `chatgpt.com`), bypassing login challenges or captcha walls.

---

#### ⚡ Data Extraction & Instant Prompt Pushing
1. **PDF Text Extraction**: The script reads the target paper locally using PyMuPDF (`fitz`), joining pages to build `full_text`.
2. **Character Limits Removed**: The script handles documents of any length, passing the full academic paper text directly to the model.
3. **Pasting the Prompt via JavaScript**: contenteditable elements (like Gemini/ChatGPT's main text input box) lag when simulating character-by-character typing for large prompts (often >50,000 characters). The script solves this by injecting the prompt *instantly* using browser JavaScript execution:
   ```javascript
   // Insert text instantly via document.execCommand into the active element
   input_el.evaluate((el, text) => {
       el.focus();
       const selection = window.getSelection();
       const range = document.createRange();
       range.selectNodeContents(el);
       selection.removeAllRanges();
       selection.addRange(range);
       document.execCommand('insertText', false, text);
   }, prompt_text);
   ```
4. **Triggering Generation**: Once pasted, the script waits for the send button to change state and clicks it:
   ```python
   send_btn.click()
   ```

---

#### 🔄 Smart Wait Loop & Text Stability Check
To handle varying response generation speeds, the script implements an active polling monitoring loop:
1. **Streaming Check**: Checks if the model is still generating by locating active loader or responder bubbles:
   `page.locator("div[aria-label='Gemini is responding'], .loading, .thinking").count()`
2. **Text Stability Tracking**: Monitors the inner text of the last message bubble:
   ```python
   current_text = els[-1].inner_text()
   ```
3. **Completion Rule**: If the text length does not change AND the streaming indicators are hidden for **6 consecutive seconds** (2 stability checks spaced 3 seconds apart), the response is determined to be complete.

---

#### ⏰ Timeout Rules & Session Recovery
To prevent script freezes, uniform **60-second timeouts** are enforced at every stage:
* **Prompt Paste Timeout**: 60s
* **Send Button Activation Timeout**: 60s
* **Response Generation Timeout**: 60s
* **Session Recovery**: If any step hangs or times out, the recovery function calls `open_new_chat(page)`. It executes `page.goto("https://gemini.google.com/app", wait_until="domcontentloaded")` and reloads the page. This starts a fresh session with a cleared context window, skipping the stuck paper and instantly continuing to the next.

---

#### 💾 Parsing and Checklist Updates
* **JSON Parsing**: The script uses a regular expression to extract the structured JSON schema enclosed within standard markdown code fences (`` ```json ... ``` ``).
* **Dataset Update**: Parses the object, validates keys (`question_id`, `question_type`, `question`, `expected_answer`, `evidence`, `difficulty`), and appends the records directly to [`gold_qa_dataset.json`](file:///Users/nanibayanaboina2750/Desktop/research-rag-assistant/papers download/gold_qa_dataset.json).
* **Checklist Update**: Reads [`papers_list.md`](file:///Users/nanibayanaboina2750/Desktop/research-rag-assistant/papers download/papers_list.md) and converts `- [ ] <paper>` to `- [x] <paper>` to track progress.

---

* **Output**: Updates [`gold_qa_dataset.json`](file:///Users/nanibayanaboina2750/Desktop/research-rag-assistant/papers download/gold_qa_dataset.json).

---

### 🔍 Phase 3: Local RAG Retrieval (Local)
* **Goal**: Search the local FAISS index for each question in the dataset and save the retrieved text/image chunks. This runs fast locally (no LLM generation).
* **Execution**:
  ```bash
  # Run retrieval for all 4,680 questions:
  python3 run_rag_evaluation.py

  # Recommended (Low-Cost/Free-Tier Limit): Run for only 50 questions:
  python3 run_rag_evaluation.py --limit 50
  ```
* **Output**: Generates [`rag_retrieved_answers.json`](file:///Users/nanibayanaboina2750/Desktop/research-rag-assistant/papers download/rag_retrieved_answers.json) containing questions, expected answers, and retrieved chunks.

---

### 💬 Phase 4: RAG Answering (Local)
* **Goal**: Generate RAG system answers based on the retrieved chunks using the Gemini API.
* **API Calls Required**: Requires exactly **1 LLM API call per processed question**. Running this on all 4,680 questions takes 4,680 API calls.
* **Execution**:
  ```bash
  # Generate answers for all retrieved questions:
  python3 run_llm_generation.py

  # Recommended (Low-Cost/Free-Tier Limit): Generate answers for only 50 questions:
  python3 run_llm_generation.py --limit 50
  ```
* **Output**: Populates the `"rag_answer"` fields inside [`rag_retrieved_answers.json`](file:///Users/nanibayanaboina2750/Desktop/research-rag-assistant/papers download/rag_retrieved_answers.json) in-place.

---

### ⚖️ Phase 5: DeepEval Judge Metric Evaluation (Local)
* **Goal**: Run Gemini as an LLM judge via `deepeval` to compare RAG answers vs. expected answers on 6 key metrics.
* **API Calls Required**: DeepEval runs multiple grading prompts under-the-hood. Grader metrics on all 4,680 questions can consume **15,000+ API calls**. Running on a subset of 50 questions keeps this under ~150 API calls (well within Gemini's free limits).
* **Execution**:
  ```bash
  # Evaluate all generated answers:
  python3 evaluate_ragvschatgpt.py

  # Recommended (Low-Cost/Free-Tier Limit): Evaluate only 50 questions:
  python3 evaluate_ragvschatgpt.py --limit 50
  ```
* **Output**: Generates [`deep_eval_results.json`](file:///Users/nanibayanaboina2750/Desktop/research-rag-assistant/papers download/deep_eval_results.json) and prints the final pipeline accuracy scores.

---

## 🏷️ File Definitions

*   [`gold_qa_dataset.json`](file:///Users/nanibayanaboina2750/Desktop/research-rag-assistant/papers download/gold_qa_dataset.json): Benchmark dataset containing generated questions, gold answers, and evidence mappings.
*   [`rag_retrieved_answers.json`](file:///Users/nanibayanaboina2750/Desktop/research-rag-assistant/papers download/rag_retrieved_answers.json): Output containing retrieved chunks and generated answers from the local RAG pipeline.
*   [`deep_eval_results.json`](file:///Users/nanibayanaboina2750/Desktop/research-rag-assistant/papers download/deep_eval_results.json): Final accuracy scores and reasons graded by DeepEval & Gemini.
*   [`gemini_judge_results.json`](file:///Users/nanibayanaboina2750/Desktop/research-rag-assistant/papers download/gemini_judge_results.json): Preserved outputs from the legacy evaluation script.

---

## 🛠️ Integration with Evaluation Frameworks

Our generated datasets are formatted to map directly to standard open-source RAG evaluation frameworks.

### 1. ⚖️ DeepEval Integration
DeepEval utilizes the `LLMTestCase` structure. You can map our output JSON files directly to test cases:

```python
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

# 1. Map our JSON data directly to DeepEval fields
test_case = LLMTestCase(
    input="What is the learning rate of the Net?",          # gold_qa_dataset.json -> "question"
    actual_output="The learning rate used was 0.01.",       # rag_retrieved_answers.json -> "rag_answer"
    expected_output="The learning rate is 0.01.",           # gold_qa_dataset.json -> "expected_answer"
    retrieval_context=[                                      # rag_retrieved_answers.json -> "retrieved_chunks"
        "We optimize parameters using SGD with a learning rate of 0.01..." 
    ]
)

# 2. Initialize and run a metric using Gemini as the judge
metric = FaithfulnessMetric(threshold=0.7, model="gemini-1.5-flash")
metric.measure(test_case)

print(f"Faithfulness Score: {metric.score}")
print(f"Feedback/Reason: {metric.reason}")
```

---

### 2. 📊 Ragas Integration
Ragas expects an Hugging Face `Dataset` object. You can load and map our JSON schemas to run Ragas evaluations:

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Map our evaluation keys into the Ragas expected schema columns
data = {
    "question": ["What is the primary contribution?"],            # "question"
    "answer": ["The primary contribution is a new Net structure."], # "rag_answer"
    "contexts": [["The paper introduces an all-convolutional Net."]],# "retrieved_chunks" text
    "ground_truth": ["An all-convolutional network layout."]       # "expected_answer"
}

dataset = Dataset.from_dict(data)
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
print(results)
```

---

### 3. 🔍 TruLens (RAG Triad)
TruLens tracks groundedness, context relevance, and answer relevance. You can feed our JSON records directly:
* **Context Relevance**: Compare `gold_qa_dataset["question"]` vs. `rag_retrieved_answers["retrieved_chunks"]`.
* **Groundedness**: Compare `rag_retrieved_answers["retrieved_chunks"]` vs. `rag_retrieved_answers["rag_answer"]`.
* **Answer Relevance**: Compare `gold_qa_dataset["question"]` vs. `rag_retrieved_answers["rag_answer"]`.

