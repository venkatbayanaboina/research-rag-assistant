"""
API-Based QA Dataset Generator
===============================
Directly queries the Gemini API using Google AI Studio to generate
15 structured QA pairs per paper, enforcing the 15 RPM free tier rate limit.

Usage:
    python3 generate_dataset_api.py
"""

import os
import re
import sys
import json
import time
from pathlib import Path
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

PDF_DIR = Path(__file__).parent / "pdfs"
CHECKLIST_PATH = Path(__file__).parent / "papers_list.md"
OUTPUT_JSON = Path(__file__).parent / "gold_qa_dataset.json"

# ── API Setup ────────────────────────────────────────────────────────────────
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("❌ GEMINI_API_KEY not found in .env. Please configure it to run.")
    sys.exit(1)

genai.configure(api_key=API_KEY)
# We use gemini-2.0-flash as the fast, high-context free tier model
MODEL_NAME = "models/gemini-2.5-flash"

# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: Path) -> str:
    """Extracts raw text from PDF file page-by-page."""
    text_blocks = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text("text")
                text_blocks.append(f"--- Page {page_num} ---\n{page_text}")
    except Exception as e:
        print(f"  ⚠️ Failed to read PDF {pdf_path.name}: {e}")
    return "\n\n".join(text_blocks)

def load_pending_papers() -> list:
    """Reads papers_list.md checklist and returns a list of pending papers (paper_id, title)."""
    pending = []
    if not CHECKLIST_PATH.exists():
        print(f"❌ Checklist file not found at: {CHECKLIST_PATH}")
        return []

    with open(CHECKLIST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            # Match lines like: - [ ] 1412.6806_Striving_for_Simplicity_The_All_Convolutional_Net.pdf - Title Here
            match = re.match(r'^\s*-\s*\[\s*\]\s*([a-zA-Z0-9\./\-_]+?)\s*-\s*(.*)$', line)
            if match:
                paper_id = match.group(1).strip()
                title = match.group(2).strip()
                pending.append((paper_id, title))
    return pending

def find_pdf_file(paper_id: str) -> Path:
    """Locates the local PDF matching the given paper ID."""
    p = PDF_DIR / paper_id
    if p.exists():
        return p
    for candidate in PDF_DIR.glob("*.pdf"):
        if paper_id in candidate.name:
            return candidate
    return None

def mark_completed(paper_id: str):
    """Updates papers_list.md checklist to check off the completed paper."""
    if CHECKLIST_PATH.exists():
        with open(CHECKLIST_PATH, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        for i, line in enumerate(lines):
            match = re.match(r'^\s*-\s*\[\s*\]\s*([a-zA-Z0-9\./\-_]+?)\s*-\s*(.*)$', line)
            if match and match.group(1).strip() == paper_id:
                title = match.group(2).strip()
                lines[i] = f"- [x] {paper_id} - {title}"
                break
        with open(CHECKLIST_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

def save_qa_pairs(paper_id: str, paper_title: str, qa_pairs: list):
    """Saves the generated QA pairs to the JSON dataset."""
    data = {}
    if OUTPUT_JSON.exists():
        try:
            with open(OUTPUT_JSON, "r") as f:
                data = json.load(f)
        except Exception:
            pass

    data[paper_id] = {
        "paper_title": paper_title,
        "qa_pairs": qa_pairs
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_json_block(text: str) -> str:
    """Extracts JSON content enclosed in markdown fences or raw body."""
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        return match.group(1)
    return text

import argparse

# ── Main Generator Loop ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="API-Based QA Dataset Generator with Sharding")
    parser.add_argument("--shard", type=int, default=1, help="Shard index to process (1-indexed)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()

    pending = load_pending_papers()
    total_all = len(pending)
    
    if total_all == 0:
        print("✅ All papers in papers_list.md are already processed!")
        return

    # Shard partition logic
    global OUTPUT_JSON
    if args.num_shards > 1:
        OUTPUT_JSON = Path(__file__).parent / f"gold_qa_dataset_shard_{args.shard}.json"
        if args.shard < 1 or args.shard > args.num_shards:
            print(f"❌ Invalid shard: {args.shard}. Must be between 1 and {args.num_shards}.")
            sys.exit(1)
        
        # Partition the list evenly
        sharded_pending = []
        for i, paper in enumerate(pending):
            if (i % args.num_shards) == (args.shard - 1):
                sharded_pending.append(paper)
        pending = sharded_pending
        print(f"分 Sharding Enabled: Running Shard {args.shard}/{args.num_shards}")
        
    total = len(pending)
    print(f"Processing batch size: {total} papers (out of {total_all} total pending).")

    if total == 0:
        print("✅ No pending papers in this shard partition!")
        return

    print(f"Initializing model: {MODEL_NAME}...")
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config={"response_mime_type": "application/json"}
    )

    RATE_LIMIT_DELAY = 4.5

    for idx, (paper_id, title) in enumerate(pending, start=1):
        print(f"\n==================================================")
        print(f"[{idx}/{total}] Processing: {paper_id}")
        print(f"Title: {title}")
        print(f"==================================================")

        pdf_path = find_pdf_file(paper_id)
        if not pdf_path:
            print(f"⚠️ PDF file not found in pdfs/ directory. Skipping.")
            continue

        t0 = time.time()
        print("Extracting text from PDF...")
        full_text = extract_pdf_text(pdf_path)
        if not full_text.strip():
            print("❌ No text extracted from PDF. Skipping.")
            continue

        num_figures = len(re.findall(r'(?i)\b(?:figure|fig\.)\b', full_text))
        num_tables = len(re.findall(r'(?i)\b(?:table)\b', full_text))
        num_equations = len(re.findall(r'(?i)\b(?:equation|eq\.)\b', full_text))
        
        fig_q = min(3, max(1, num_figures // 3)) if num_figures > 0 else 0
        tbl_q = min(3, max(1, num_tables // 3)) if num_tables > 0 else 0
        eq_q = min(2, max(1, num_equations // 3)) if num_equations > 0 else 0
        text_q = 15 - (fig_q + tbl_q + eq_q)

        prompt_text = f"""
You are an expert AI academic evaluator. Your task is to analyze the academic paper text provided below and generate exactly 15 high-quality, diverse, and research-grade Question-Answer (QA) pairs for evaluating a Retrieval-Augmented Generation (RAG) assistant.

Generate exactly 15 QA pairs according to these constraints:
- Generate {text_q} text-based QA pairs (evaluating comprehension of methodologies, results, related work).
- Generate {fig_q} figure-based QA pairs (asking about results, trends, charts mentioned as Figure X).
- Generate {tbl_q} table-based QA pairs (asking about parameters, comparisons, or data inside Table Y).
- Generate {eq_q} equation-based QA pairs (asking about variables, losses, or functions inside Equation Z).

Each QA pair must have:
1. `question_id`: uniquely named string format: '{paper_id}_Q01' to '{paper_id}_Q15'
2. `question_type`: exactly 'text', 'figure', 'table', or 'equation'
3. `question`: clear, descriptive, self-contained question. Do not refer to "the paper" or "this text", instead cite the specific paper or method title where possible.
4. `expected_answer`: detailed, scientifically rigorous answer covering the details from the text.
5. `difficulty`: exactly 'easy', 'medium', or 'hard' (aim for 5 easy, 5 medium, 5 hard).
6. `evidence`: An object mapping where the answer is found in the text:
   - `page`: (mandatory) integer page number where the fact resides.
   - `section`: (optional) section heading name.
   - `paragraph`: (optional) paragraph index or count on that page.
   - `figure`: e.g. 'Figure 3' if type is 'figure'.
   - `table`: e.g. 'Table 1' if type is 'table'.
   - `equation`: e.g. 'Equation (5)' if type is 'equation'.

Only cite figures, tables, and equations that actually appear in the text. Do not invent figure, table, or equation numbers.

You MUST output a JSON object conforming exactly to this JSON Schema:
{{
  "paper_id": "{paper_id}",
  "paper_title": "{title}",
  "qa_pairs": [
    {{
      "question_id": "string",
      "question_type": "string",
      "question": "string",
      "expected_answer": "string",
      "evidence": {{
        "page": 1,
        "section": "string",
        "paragraph": 1,
        "figure": "string",
        "table": "string",
        "equation": "string"
      }},
      "difficulty": "string"
    }}
  ]
}}

Here is the full text of the academic paper for your analysis:

[PAPER TEXT START]
{full_text}
[PAPER TEXT END]
"""

        print(f"Calling Gemini API ({MODEL_NAME})...")
        api_start = time.time()
        try:
            response = model.generate_content(prompt_text)
            response_text = response.text
            api_elapsed = time.time() - api_start
            
            json_str = extract_json_block(response_text)
            parsed_data = json.loads(json_str)
            
            qa_pairs = parsed_data.get("qa_pairs", [])
            print(f"  ✓ Successfully generated {len(qa_pairs)} QA pairs in {round(api_elapsed, 1)}s")
            
            save_qa_pairs(paper_id, title, qa_pairs)
            mark_completed(paper_id)
            print("  ✓ Saved to gold_qa_dataset.json and checked off in papers_list.md.")

        except Exception as e:
            print(f"  ❌ API Generation failed for {paper_id}: {e}")

        elapsed_total = time.time() - t0
        sleep_needed = max(0.1, RATE_LIMIT_DELAY - elapsed_total)
        if idx < total:
            print(f"Waiting {round(sleep_needed, 1)}s to maintain free-tier rate limits...")
            time.sleep(sleep_needed)

    print("\n🎉 Gold QA Dataset generation completed successfully!")

if __name__ == "__main__":
    main()
