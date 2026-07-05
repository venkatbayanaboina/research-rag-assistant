# 📚 Multimodal QA Evaluation Dataset Generation Procedure

This document outlines the procedure for generating 15 structured QA pairs per paper to evaluate the Retrieval-Augmented Generation (RAG) assistant.

To bypass API pricing, usage limits, and transient error codes (e.g., OpenRouter `402 Payment Required`), this process leverages browser automation (`automate_chatgpt.py`) to upload papers and extract generated JSON responses directly from the ChatGPT web interface.

---

## 📊 Context & Statistics
Based on a complete local scan of the remaining uncompleted PDFs in the dataset:
* **Total Scanned PDFs**: 276
* **Average Words per Paper**: ~7,730 words
* **Average Tokens per Paper (Estimated)**: ~10,281 tokens (ratio of 1.33 tokens per word)

### Session Refresh Strategy
To prevent context window bloat, response sluggishness, and cross-paper details mixing (which leads to hallucinated page or table numbers), the script implements an **automatic session refresh**:
* **Optimal threshold**: **Every 4 papers** ($4 \times 10,281 \approx 41,124$ tokens).
* After every 4 papers, the automation will automatically reload the base URL `https://chatgpt.com` to start a brand new chat session with a clean context.

---

## 🛠️ Step-by-Step Setup

### 1. Install Dependencies
Ensure the necessary libraries and browser binaries are installed in your Python environment:
```bash
pip install playwright pymupdf
playwright install
```

### 2. Launch Google Chrome in Debug Mode
Close all standard Chrome windows, then run the following command in your terminal to launch Chrome with Remote Debugging enabled on port `9222`:
```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 --user-data-dir="/Users/nanibayanaboina2750/ChromeProfile"
```

### 3. Log In to ChatGPT
In the Chrome window launched by the step above:
1. Go to [https://chatgpt.com](https://chatgpt.com).
2. Log in manually using your credentials.
3. Keep this window open.

### 4. Execute the Automation
In your terminal, navigate to the `papers download` directory and run the script:
```bash
cd "/Users/nanibayanaboina2750/Desktop/research-rag-assistant/papers download"
python3 automate_chatgpt.py
```

---

## ⚙️ How the Automation Works

1. **Checklist Scanning**: The script reads [papers_list.md](file:///Users/nanibayanaboina2750/Desktop/research-rag-assistant/papers%20download/papers_list.md) to locate the next unprocessed (`- [ ]`) paper.
2. **Dynamic Modal Allocation**: It scans the PDF text locally to detect figures, tables, and equations, and calculates the exact counts required for the 15 QA pairs.
3. **File Upload**: It uploads the PDF to the input file input (`#upload-files`) and waits up to **30 seconds** for the upload to complete.
4. **Prompt Submission**: It fills in the detailed system schema prompt and submits it once the Send button becomes active.
5. **Smart Wait Loop**: It enters a monitoring loop that polls the browser tab:
   - Checks if the stop-generating button disappears.
   - Monitors when the send button becomes visible again.
   - Fallback: Checks the text of the last response element to verify it has stopped growing (remains stable for 6 seconds).
6. **JSON Extraction & Sanitization**: It extracts the JSON block from the message bubble (with a global body-text fallback), parses it, appends it to [evaluation_dataset.json](file:///Users/nanibayanaboina2750/Desktop/research-rag-assistant/papers%20download/evaluation_dataset.json), and checks the paper off in the checklist.
