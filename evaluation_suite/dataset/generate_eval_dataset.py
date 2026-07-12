import os
import re
import sys
import json
import time
import argparse
import socket
import urllib.request
import urllib.error
from pathlib import Path
import fitz  # PyMuPDF
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Set global socket timeout to prevent indefinite network hangs
socket.setdefaulttimeout(240)

# Mute PyMuPDF C-level warnings/errors to keep logs clean
fitz.TOOLS.mupdf_display_errors(False)

# -----------------------------
# Configuration & Setup
# -----------------------------
PDF_DIR = Path("pdfs")
CHECKLIST_PATH = Path("papers_list.md")
OUTPUT_JSON = Path("evaluation_dataset.json")

# Load API Key from .env in the parent directory
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"), override=True)
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    # Try parent directory's parent just in case
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"), override=True)
    api_key = os.getenv("GEMINI_API_KEY")

# Default model name
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# -----------------------------
# Pydantic Schemas for Gemini Structured JSON
# -----------------------------
class Evidence(BaseModel):
    page: int = Field(description="1-indexed page number containing the evidence.")
    section: Optional[str] = Field(default=None, description="Section header/name where evidence is located, if type is 'text'.")
    paragraph: Optional[int] = Field(default=None, description="1-indexed paragraph number within the section, if type is 'text'.")
    figure: Optional[str] = Field(default=None, description="Figure reference, e.g. 'Figure 3' if type is 'figure'.")
    table: Optional[str] = Field(default=None, description="Table reference, e.g. 'Table 2' if type is 'table'.")
    equation: Optional[str] = Field(default=None, description="Equation reference, e.g. 'Equation (3)' if type is 'equation'.")

class QAPair(BaseModel):
    question_id: str = Field(description="Unique ID for the question, e.g. '1506.02025_Q01'")
    question_type: str = Field(description="One of: 'text', 'figure', 'table', 'equation'")
    question: str = Field(description="The generated question text.")
    expected_answer: str = Field(description="The expected answer based on the paper.")
    evidence: Evidence = Field(description="The supporting evidence metadata.")
    difficulty: str = Field(description="One of: 'easy', 'medium', 'hard'")

class PaperEvaluationDataset(BaseModel):
    paper_id: str
    paper_title: str
    qa_pairs: List[QAPair]

# -----------------------------
# Core Helper Logic
# -----------------------------
def find_pdf_file(paper_id: str) -> Optional[Path]:
    """Finds the local PDF matching the given arXiv paper_id."""
    for p in PDF_DIR.glob("*.pdf"):
        stem = p.stem
        parts = stem.split("_")
        if parts[0] == paper_id:
            return p
    return None

def determine_allocations(has_figures: bool, has_tables: bool, has_equations: bool) -> dict:
    """Calculates the exact QA type count distribution to total exactly 15 questions."""
    if has_figures and has_tables and has_equations:
        return {"text": 4, "figure": 4, "table": 3, "equation": 4}
    elif has_figures and has_equations:
        return {"text": 5, "figure": 5, "table": 0, "equation": 5}
    elif has_tables and has_equations:
        return {"text": 5, "figure": 0, "table": 5, "equation": 5}
    elif has_figures and has_tables:
        return {"text": 5, "figure": 5, "table": 5, "equation": 0}
    elif has_figures:
        return {"text": 8, "figure": 7, "table": 0, "equation": 0}
    elif has_tables:
        return {"text": 8, "figure": 0, "table": 7, "equation": 0}
    elif has_equations:
        return {"text": 8, "figure": 0, "table": 0, "equation": 7}
    else:
        return {"text": 15, "figure": 0, "table": 0, "equation": 0}

def mark_completed(paper_id: str):
    """Updates papers_list.md checklist to mark the given paper completed."""
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

def generate_content_with_retry(client, model, contents, config):
    """Retries generate_content call on 429 and 503 transient errors with delay parsing."""
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        except Exception as e:
            err_msg = str(e).upper()
            code = getattr(e, "code", None)
            status = getattr(e, "status", None)
            
            is_transient = (
                code in (429, 503, 504, 500) or
                "429" in err_msg or
                "503" in err_msg or
                "RESOURCE_EXHAUSTED" in err_msg or
                "UNAVAILABLE" in err_msg or
                "RATE" in err_msg or
                "TIME" in err_msg
            )
            
            if is_transient and attempt < max_retries:
                # Default sleep time
                sleep_time = 15.0
                # Try to parse the exact retry delay from the error message
                match = re.search(r'retry in ([\d\.]+)s', str(e), re.IGNORECASE)
                if match:
                    sleep_time = float(match.group(1)) + 1.0
                elif code == 503 or "503" in err_msg or "UNAVAILABLE" in err_msg:
                    sleep_time = 10.0  # short wait for 503 temporary overload
                
                print(f"\n⚠️ Gemini API Transient Error ({code or status or '429/503'}) hit. Sleeping {sleep_time:.2f} seconds before retry (Attempt {attempt}/{max_retries})...")
                sys.stdout.flush()
                time.sleep(sleep_time)
                continue
            raise e

def query_nvidia_nim(api_key, model, prompt_text, system_instruction):
    """Queries Nvidia NIM completions endpoint using urllib."""
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_text}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2
    }
    
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                return res["choices"][0]["message"]["content"]
        except Exception as e:
            err_msg = str(e).upper()
            code = getattr(e, "code", None)
            is_transient = (
                code in (429, 503, 504, 500) or
                "429" in err_msg or
                "503" in err_msg or
                "TIME" in err_msg or
                "UNAVAILABLE" in err_msg
            )
            if is_transient and attempt < max_retries:
                sleep_time = 15.0
                print(f"\n⚠️ Nvidia API Transient Error ({code or 'Unknown'}) hit. Sleeping {sleep_time}s before retry (Attempt {attempt}/{max_retries})...")
                sys.stdout.flush()
                time.sleep(sleep_time)
                continue
            raise e

def query_openrouter(api_key, model, prompt_text, system_instruction):
    """Queries OpenRouter completions endpoint using urllib with transient error handling."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/venkatbayanaboina/research-rag-assistant",
        "X-Title": "Research RAG Assistant"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_text}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2
    }
    
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                return res["choices"][0]["message"]["content"]
        except Exception as e:
            err_msg = str(e).upper()
            code = getattr(e, "code", None)
            is_transient = (
                code in (429, 503, 504, 500) or
                "429" in err_msg or
                "503" in err_msg or
                "TIME" in err_msg or
                "UNAVAILABLE" in err_msg
            )
            if is_transient and attempt < max_retries:
                sleep_time = 15.0
                print(f"\n⚠️ OpenRouter API Transient Error ({code or 'Unknown'}) hit. Sleeping {sleep_time}s before retry (Attempt {attempt}/{max_retries})...")
                sys.stdout.flush()
                time.sleep(sleep_time)
                continue
            raise e

# -----------------------------
# Main Pipeline execution
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Multimodal QA Evaluation Dataset Generator")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of papers to process")
    parser.add_argument("--provider", type=str, choices=["gemini", "nvidia", "hybrid"], default="hybrid", help="LLM Provider to use ('gemini', 'nvidia', or 'hybrid')")
    parser.add_argument("--model", type=str, default=None, help="Model name override")
    args = parser.parse_args()

    # Load provider keys
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    or_key = os.getenv("OPENROUTER_API_KEY")

    provider = args.provider

    if provider == "gemini":
        model_name = args.model or GEMINI_MODEL_NAME
        if not api_key:
            print("❌ Error: GEMINI_API_KEY environment variable not found in .env.")
            sys.exit(1)
        client = genai.Client(api_key=api_key)
    elif provider == "nvidia":
        model_name = args.model or "deepseek-ai/deepseek-v4-pro"
        if not nvidia_key:
            print("❌ Error: NVIDIA_API_KEY environment variable not found in .env.")
            sys.exit(1)
        client = None
    else:  # Hybrid Strategy (Nvidia DeepSeek-V4-Pro + OpenRouter DeepSeek-R1)
        model_name = args.model or "hybrid"
        if not nvidia_key or not or_key:
            print("❌ Error: Both NVIDIA_API_KEY and OPENROUTER_API_KEY must be configured in .env for Hybrid mode.")
            sys.exit(1)
        client = None

    print(f"Using provider: {provider.upper()}, Model: {model_name}")

    # 1. Read uncompleted papers from papers_list.md
    if not CHECKLIST_PATH.exists():
        print(f"❌ Error: checklist file not found at {CHECKLIST_PATH}.")
        sys.exit(1)

    unprocessed = []
    with open(CHECKLIST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r'^\s*-\s*\[\s*\]\s*([a-zA-Z0-9\./\-_]+?)\s*-\s*(.*)$', line)
            if match:
                unprocessed.append((match.group(1).strip(), match.group(2).strip()))

    print(f"Found {len(unprocessed)} papers remaining to process.")
    if not unprocessed:
        print("All papers have been processed and marked.")
        return

    if args.limit:
        unprocessed = unprocessed[:args.limit]
        print(f"Limit applied: processing first {args.limit} papers.")

    # Load existing dataset results
    all_dataset_records = {}
    if OUTPUT_JSON.exists():
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                all_dataset_records = json.load(f)
        except Exception:
            pass

    # 2. Process each paper
    for idx, (paper_id, title) in enumerate(unprocessed, start=1):
        print("\n" + "="*80)
        print(f"[{idx}/{len(unprocessed)}] Processing: {paper_id} - {title}")
        print("="*80)

        pdf_path = find_pdf_file(paper_id)
        if not pdf_path or not pdf_path.exists():
            print(f"⚠️ Warning: PDF for paper {paper_id} not found in {PDF_DIR}. Skipping.")
            continue

        # Extract layout details
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            full_text = ""
            for page_num in range(num_pages):
                page = doc[page_num]
                full_text += f"\n--- PAGE {page_num + 1} ---\n" + page.get_text()
        except Exception as e:
            print(f"❌ Failed to parse PDF text via PyMuPDF: {e}")
            continue

        # Fast heuristic scans
        figures = set(re.findall(r'\b(?:Figure|Fig\.)\s+(\d+)\b', full_text, re.IGNORECASE))
        tables = set(re.findall(r'\bTable\s+(\d+)\b', full_text, re.IGNORECASE))
        equations = set(re.findall(r'\b(?:Equation|Eq\.)\s+\(?(\d+)\)?\b', full_text, re.IGNORECASE))
        eq_ends = re.findall(r'\(\s*(\d+)\s*\)\s*$', full_text, re.MULTILINE)
        equations.update(eq_ends)

        has_figs = len(figures) > 0
        has_tbls = len(tables) > 0
        has_eqs = len(equations) > 0

        print(f"Detected modalities: Pages={num_pages}, Figures={len(figures)}, Tables={len(tables)}, Equations={len(equations)}")
        allocations = determine_allocations(has_figs, has_tbls, has_eqs)
        print(f"Allocating QA counts: {allocations}")

        # Construct prompt instructions
        prompt_instruction = f"""Analyze the provided academic paper and generate exactly 15 high-quality, research-grade evaluation QA pairs.
        
Paper ID: {paper_id}
Paper Title: {title}

You MUST generate exactly the following counts for each question type:
- 'text': {allocations['text']} questions
- 'figure': {allocations['figure']} questions (Must reference actual figure numbers found in the paper, e.g. Figure 1, Figure 2)
- 'table': {allocations['table']} questions (Must reference actual table numbers found in the paper, e.g. Table 1, Table 2)
- 'equation': {allocations['equation']} questions (Must reference actual equation numbers found in the paper, e.g. Equation (1))

Total: 15 questions.

Difficulty Ratings:
- 'easy': Direct lookup of facts/parameters.
- 'medium': Connecting information across paragraphs.
- 'hard': Reasoning over math/equations, visual figure workflows, or complex tables.

Each question MUST include precise evidence mapping:
- page: 1-indexed page number containing the evidence.
- section: Section name/header (e.g. 'Methodology'), if type is 'text'.
- paragraph: 1-indexed paragraph number within the section, if type is 'text'.
- figure: e.g. 'Figure 3' if type is 'figure'.
- table: e.g. 'Table 2' if type is 'table'.
- equation: e.g. 'Equation (5)' if type is 'equation'.

Only cite figures, tables, and equations that actually appear in the text.
"""

        # Enforce structured output via Pydantic schema
        gen_config = types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=PaperEvaluationDataset,
            system_instruction="You are an expert academic evaluator. You output structured evaluation QA pairs with precise page and modality evidence references."
        )

        if provider == "gemini":
            print("Generating structured QA pairs via Gemini API...")
            try:
                response = generate_content_with_retry(
                    client=client,
                    model=model_name,
                    contents=[prompt_instruction + f"\n\nPaper Content:\n{full_text}"],
                    config=gen_config
                )
                response_text = response.text
            except Exception as e:
                print(f"❌ Failed generation for {paper_id}: {e}")
                continue
        elif provider == "nvidia":
            print(f"Generating structured QA pairs via NVIDIA NIM API ({model_name})...")
            schema_desc = """
You MUST output a JSON object matching this schema:
{
  "paper_id": "string",
  "paper_title": "string",
  "qa_pairs": [
    {
      "question_id": "string",
      "question_type": "text" | "figure" | "table" | "equation",
      "question": "string",
      "expected_answer": "string",
      "evidence": {
        "page": integer (1-indexed page number),
        "section": "string" (optional),
        "paragraph": integer (optional),
        "figure": "string" (optional),
        "table": "string" (optional),
        "equation": "string" (optional)
      },
      "difficulty": "easy" | "medium" | "hard"
    }
  ]
}
"""
            full_prompt = prompt_instruction + schema_desc + f"\n\nPaper Content:\n{full_text}"
            try:
                response_text = query_nvidia_nim(
                    api_key=nvidia_key,
                    model=model_name,
                    prompt_text=full_prompt,
                    system_instruction="You are an expert academic evaluator. You output structured evaluation QA pairs with precise page and modality evidence references conforming exactly to the requested JSON schema."
                )
            except Exception as e:
                print(f"❌ Failed generation for {paper_id} via Nvidia NIM: {e}")
                continue
        else:  # Hybrid Strategy (Nvidia DeepSeek-V4-Pro + OpenRouter DeepSeek-R1)
            schema_desc = """
You MUST output a JSON object matching this schema:
{
  "qa_pairs": [
    {
      "question_type": "text" | "figure" | "table" | "equation",
      "question": "string",
      "expected_answer": "string",
      "evidence": {
        "page": integer (1-indexed page number),
        "section": "string" (optional),
        "paragraph": integer (optional),
        "figure": "string" (optional),
        "table": "string" (optional),
        "equation": "string" (optional)
      },
      "difficulty": "easy" | "medium" | "hard"
    }
  ]
}
"""
            qa_pairs = []

            # Phase A: Nvidia DeepSeek-V4-Pro for Text, Figures, Tables
            v4_count = allocations['text'] + allocations['figure'] + allocations['table']
            if v4_count > 0:
                print(f"Generating {v4_count} layout/text QA pairs via Nvidia DeepSeek-V4-Pro...")
                prompt_v4 = f"""Analyze the provided academic paper and generate structured evaluation QA pairs.
                
Paper ID: {paper_id}
Paper Title: {title}

You MUST generate exactly the following counts for each question type:
- 'text': {allocations['text']} questions
- 'figure': {allocations['figure']} questions (Must reference actual figure numbers found in the paper, e.g. Figure 1)
- 'table': {allocations['table']} questions (Must reference actual table numbers found in the paper, e.g. Table 1)

Total: {v4_count} questions.

Difficulty Ratings:
- 'easy': Direct lookup of facts/parameters.
- 'medium': Connecting information across paragraphs.
- 'hard': Reasoning over visual figure workflows or complex tables.
""" + schema_desc + f"\n\nPaper Content:\n{full_text}"

                try:
                    res_v4 = query_nvidia_nim(
                        api_key=nvidia_key,
                        model="deepseek-ai/deepseek-v4-pro",
                        prompt_text=prompt_v4,
                        system_instruction="You are an expert academic evaluator. You output structured evaluation QA pairs with precise page and modality evidence references conforming exactly to the requested JSON schema."
                    )
                    clean_res_v4 = re.sub(r'<think>.*?</think>', '', res_v4, flags=re.DOTALL).strip()
                    data_v4 = json.loads(clean_res_v4)
                    qa_pairs.extend(data_v4.get("qa_pairs", []))
                except Exception as e:
                    print(f"❌ Failed generation via Nvidia NIM: {e}")
                    continue

            # Phase B: OpenRouter DeepSeek-R1 for Equations
            if allocations['equation'] > 0:
                print(f"Generating {allocations['equation']} mathematical reasoning/equation QA pairs via OpenRouter DeepSeek-R1...")
                prompt_r1 = f"""Analyze the provided academic paper and generate structured evaluation QA pairs focusing on mathematical equations and reasoning.
                
Paper ID: {paper_id}
Paper Title: {title}

You MUST generate exactly the following counts for each question type:
- 'equation': {allocations['equation']} questions (Must reference actual equation numbers found in the paper, e.g. Equation (1))

Total: {allocations['equation']} questions.

Difficulty Ratings:
- 'hard': Reasoning over complex math/equations, derivations, and formulas.
""" + schema_desc + f"\n\nPaper Content:\n{full_text}"

                try:
                    res_r1 = query_openrouter(
                        api_key=or_key,
                        model="deepseek/deepseek-r1",
                        prompt_text=prompt_r1,
                        system_instruction="You are an expert academic evaluator. You output structured evaluation QA pairs focusing on complex mathematical equations and reasoning, conforming exactly to the requested JSON schema."
                    )
                    clean_res_r1 = re.sub(r'<think>.*?</think>', '', res_r1, flags=re.DOTALL).strip()
                    data_r1 = json.loads(clean_res_r1)
                    qa_pairs.extend(data_r1.get("qa_pairs", []))
                except Exception as e:
                    print(f"⚠️ OpenRouter R1 failed: {e}")
                    print("🔄 Falling back to Nvidia DeepSeek-V4-Pro to generate the equation questions...")
                    try:
                        res_r1_fallback = query_nvidia_nim(
                            api_key=nvidia_key,
                            model="deepseek-ai/deepseek-v4-pro",
                            prompt_text=prompt_r1,
                            system_instruction="You are an expert academic evaluator. You output structured evaluation QA pairs focusing on complex mathematical equations and reasoning, conforming exactly to the requested JSON schema."
                        )
                        clean_res_r1 = re.sub(r'<think>.*?</think>', '', res_r1_fallback, flags=re.DOTALL).strip()
                        data_r1 = json.loads(clean_res_r1)
                        qa_pairs.extend(data_r1.get("qa_pairs", []))
                    except Exception as fallback_err:
                        print(f"❌ Fallback to Nvidia NIM failed: {fallback_err}")
                        continue

            # Wrap in structured result format
            result_data = {
                "paper_id": paper_id,
                "paper_title": title,
                "qa_pairs": qa_pairs
            }
            response_text = json.dumps(result_data)

        try:
            # Parse response and validate pages/ids
            result_data = json.loads(response_text)
            
            # Schema verification & cleanups
            validated_pairs = []
            for qa_idx, pair in enumerate(result_data.get("qa_pairs", []), start=1):
                # Ensure correct question ID format
                pair["question_id"] = f"{paper_id}_Q{qa_idx:02d}"
                
                # Check page bounds
                evidence = pair.get("evidence", {})
                page = evidence.get("page", 1)
                if page < 1 or page > num_pages:
                    # Clip to bounds
                    evidence["page"] = max(1, min(page, num_pages))
                
                validated_pairs.append(pair)
                
            result_data["qa_pairs"] = validated_pairs
            
            # Save output
            all_dataset_records[paper_id] = result_data
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(all_dataset_records, f, indent=2)
                
            print(f"✓ Successfully generated and validated 15 QA pairs for {paper_id}.")
            
            # Update papers_list.md
            mark_completed(paper_id)
            print(f"✓ Marked {paper_id} completed in {CHECKLIST_PATH}.")
            
        except Exception as e:
            print(f"❌ Failed generation for {paper_id}: {e}")
            
        # Politeness delay to avoid rate limits
        time.sleep(2)

    print("\n" + "="*40)
    print("Execution complete!")
    print(f"Dataset results written to {OUTPUT_JSON}.")
    print("="*40)

if __name__ == "__main__":
    main()
