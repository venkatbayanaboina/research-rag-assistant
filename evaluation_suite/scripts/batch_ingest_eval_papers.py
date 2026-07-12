"""
Batch Ingest Evaluation Papers
==============================
Reads evaluation_dataset.json to find all the papers we are evaluating,
checks if they are already indexed in the FAISS store, and if not,
indexes them automatically using the fast strategy.

Usage:
    python3 batch_ingest_eval_papers.py
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

# ── Make sure root imports work ──────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.core.vector_store import get_indexed_documents, add_document_to_store
from src.core.ingestion import parse_pdf
from src.core.chunker import process_text_chunks, process_image_chunks

# ── Paths ────────────────────────────────────────────────────────────────────
EVAL_DATASET  = ROOT / "evaluation_suite" / "gold_qa_dataset.json"
PDF_DIR       = ROOT / "evaluation_suite" / "pdfs"

def find_pdf_file(paper_id: str) -> Path:
    """Finds the local PDF matching the given arXiv paper_id."""
    for p in PDF_DIR.glob("*.pdf"):
        stem = p.name
        parts = stem.split("_")
        if parts[0] == paper_id:
            return p
    return None

import argparse

def main():
    parser = argparse.ArgumentParser(description="Batch Ingest Evaluation Papers")
    parser.add_argument("--strategy", type=str, choices=["fast", "hi_res"], default="hi_res",
                        help="Ingestion strategy: fast (text-only) or hi_res (multimodal text + diagrams)")
    args = parser.parse_args()

    if not EVAL_DATASET.exists():
        print(f"❌ {EVAL_DATASET} not found.")
        sys.exit(1)

    with open(EVAL_DATASET) as f:
        dataset = json.load(f)

    # 1. Get already indexed files (normalized to basenames)
    indexed_files = {os.path.basename(f) for f in get_indexed_documents()}
    print(f"Currently indexed files in FAISS: {len(indexed_files)}")
    print(f"Ingestion mode strategy: {args.strategy}")

    # 2. Find papers in eval dataset that are not indexed
    papers_to_ingest = []
    for paper_id in dataset.keys():
        pdf_path = find_pdf_file(paper_id)
        if not pdf_path:
            print(f"⚠️ PDF for paper ID {paper_id} not found in {PDF_DIR.name}/")
            continue
        
        if pdf_path.name not in indexed_files:
            papers_to_ingest.append((paper_id, pdf_path))

    total = len(papers_to_ingest)
    print(f"Total papers to ingest: {total}")

    if total == 0:
        print("✅ All evaluation papers are already indexed in FAISS!")
        return

    print("\nStarting ingestion process...")
    for idx, (paper_id, pdf_path) in enumerate(papers_to_ingest, start=1):
        print(f"\n==================================================")
        print(f"[{idx}/{total}] Ingesting: {pdf_path.name}")
        print(f"==================================================")
        
        t0 = time.time()
        try:
            # 1. Parse PDF using selected strategy
            elements = parse_pdf(str(pdf_path), strategy=args.strategy)
            
            # 2. Extract chunks
            text_chunks = process_text_chunks(elements, str(pdf_path))
            image_chunks = process_image_chunks(elements, str(pdf_path)) if args.strategy == "hi_res" else []
            
            # 3. Add to FAISS store
            add_document_to_store(text_chunks, image_chunks)
            
            print(f"✓ Completed in {round(time.time() - t0, 1)}s. Chunks: Text={len(text_chunks)}, Images={len(image_chunks)}")
        except Exception as e:
            print(f"❌ Failed to ingest {pdf_path.name}: {e}")
            
        time.sleep(1)

    print("\n✅ Batch ingestion completed successfully!")

if __name__ == "__main__":
    main()
