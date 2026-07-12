"""
ingest_sample_papers.py
-------------------------
1. Reads sample_100_answers.json
2. Finds the unique papers needed for these 100 questions
3. Ingests ONLY those papers into FAISS using the fast (text-only) strategy
"""

import os
import sys
import json
import time
from pathlib import Path

# Disable parallel forks to avoid segmentation faults on macOS
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.core.vector_store import get_indexed_documents, add_document_to_store
from src.core.ingestion import parse_pdf
from src.core.chunker import process_text_chunks

SAMPLE_IN   = ROOT / "evaluation_suite" / "sample_100_answers.json"
PDF_DIR     = ROOT / "evaluation_suite" / "pdfs"

def find_pdf_file(paper_id: str) -> Path:
    for p in PDF_DIR.glob("*.pdf"):
        if p.name.split("_")[0] == paper_id:
            return p
    return None

def main():
    if not SAMPLE_IN.exists():
        print(f"❌ {SAMPLE_IN} not found. Run sample_and_generate.py first.")
        sys.exit(1)

    with open(SAMPLE_IN) as f:
        sample = json.load(f)

    # 1. Identify papers needed
    needed_paper_ids = set()
    for qid in sample.keys():
        needed_paper_ids.add(qid.split("_")[0])

    print(f"📋 Needed papers for 100 questions: {len(needed_paper_ids)}")

    # 2. Get currently indexed files
    indexed_files = {os.path.basename(f) for f in get_indexed_documents()}
    print(f"Currently indexed files: {len(indexed_files)}")

    # 3. Find files to ingest
    to_ingest = []
    for pid in needed_paper_ids:
        pdf_path = find_pdf_file(pid)
        if not pdf_path:
            print(f"⚠️ PDF for paper ID {pid} not found!")
            continue
        if pdf_path.name not in indexed_files:
            to_ingest.append(pdf_path)

    print(f"⏳ Papers to ingest: {len(to_ingest)}")

    if not to_ingest:
        print("✅ All needed papers are already indexed!")
        return

    # 4. Ingest sequentially
    for idx, pdf_path in enumerate(to_ingest, 1):
        print(f"\n[{idx}/{len(to_ingest)}] Ingesting: {pdf_path.name}")
        t0 = time.time()
        try:
            elements = parse_pdf(str(pdf_path), strategy="fast")
            text_chunks = process_text_chunks(elements, str(pdf_path))
            add_document_to_store(text_chunks, [])
            print(f"✓ Done in {time.time() - t0:.1f}s | Chunks: {len(text_chunks)}")
        except Exception as e:
            print(f"❌ Failed: {e}")
        time.sleep(1)

    print("\n✅ Ingestion complete!")

if __name__ == "__main__":
    main()
