import os
import re
import sys
import json
import time
import numpy as np
import fitz  # PyMuPDF
import faiss
from pathlib import Path
from unstructured.partition.pdf import partition_pdf

# Set workspace paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.core.embedder import embed_clip_text, embed_image_batch

# Mute PyMuPDF C-level warnings/errors to keep logs clean
fitz.TOOLS.mupdf_display_errors(False)

# Ground Truth definition
GROUND_TRUTH = {
    "1408.5882": {
        "title": "Convolutional Neural Networks for Sentence Classification",
        "filename": "1408.5882_Convolutional_Neural_Networks_for_Sentence_Classification.pdf",
        "figures": [2],       # Figure 1 on Page 2
        "tables": [3, 4, 4]   # Table 1 on Page 3, Table 2 on Page 4, Table 3 on Page 4
    },
    "1407.7906": {
        "title": "How Auto-Encoders Could Provide Credit Assignment...",
        "filename": "1407.7906_How_Auto-Encoders_Could_Provide_Credit_Assignment_in_Deep_Networks_via_Target_Propagation.pdf",
        "figures": [5, 10, 11, 13, 13],  # Fig 1: p5, Fig 2: p10, Fig 3: p11, Fig 4: p13, Fig 5: p13
        "tables": []
    },
    "1505.05424": {
        "title": "Weight Uncertainty in Neural Networks",
        "filename": "1505.05424_Weight_Uncertainty_in_Neural_Networks.pdf",
        "figures": [4, 6, 7, 8, 9, 9],    # Fig 1: p4, Fig 2: p6, Fig 3: p7, Fig 4: p8, Fig 5: p9, Fig 6: p9
        "tables": [8, 8]                  # Table 1: p8, Table 2: p8
    }
}

PDF_DIR = Path("papers download/pdfs")
EVALUATION_REPORT_PATH = Path(".gemini/antigravity/brain/155c5f43-9069-4a7d-84ae-838bc33963b4/evaluation_what_we_are_doing.md")

def calculate_metrics(gold_pages, predicted_pages, total_pages):
    """Calculates page-level Precision, Recall, and F1-Score."""
    gold_set = set(gold_pages)
    pred_set = set(predicted_pages)
    
    tp = len(gold_set.intersection(pred_set))
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def run_layout_evaluation():
    print("\n" + "="*80)
    # Phase 1: Ingestion / Layout extraction comparison
    print("🚀 PHASE 1: EVALUATING LAYOUT DETECTION (YOLOX VS PyMuPDF)")
    print("="*80)
    
    results = {
        "yolox": {"figures": [], "tables": [], "latencies": []},
        "pymupdf": {"figures": [], "tables": [], "latencies": []}
    }
    
    for paper_id, gt in GROUND_TRUTH.items():
        pdf_path = PDF_DIR / gt["filename"]
        if not pdf_path.exists():
            print(f"⚠️ Warning: PDF for paper {paper_id} not found at {pdf_path}. Skipping.")
            continue
            
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        
        # --- PYMUPDF EVALUATION ---
        start_time = time.time()
        pred_figs_py = []
        pred_tbls_py = []
        
        for p_idx in range(num_pages):
            page = doc[p_idx]
            page_num = p_idx + 1
            
            # Detect images using PyMuPDF standard get_images
            if len(page.get_images()) > 0:
                pred_figs_py.append(page_num)
                
            # Detect tables using find_tables
            try:
                tables = page.find_tables()
                if len(tables) > 0:
                    pred_tbls_py.append(page_num)
            except AttributeError:
                # Fallback if find_tables isn't supported on old version
                pass
                
        pymupdf_latency = (time.time() - start_time) * 1000  # ms
        results["pymupdf"]["latencies"].append(pymupdf_latency)
        
        py_fig_p, py_fig_r, py_fig_f = calculate_metrics(gt["figures"], pred_figs_py, num_pages)
        py_tbl_p, py_tbl_r, py_tbl_f = calculate_metrics(gt["tables"], pred_tbls_py, num_pages)
        
        results["pymupdf"]["figures"].append((py_fig_p, py_fig_r, py_fig_f))
        results["pymupdf"]["tables"].append((py_tbl_p, py_tbl_r, py_tbl_f))
        
        # --- YOLOX EVALUATION ---
        start_time = time.time()
        pred_figs_yx = []
        pred_tbls_yx = []
        
        try:
            # Run Poppler/YOLOX strategy
            elements = partition_pdf(
                filename=str(pdf_path),
                strategy="hi_res",
                hi_res_model_name="yolox"
            )
            for el in elements:
                page_num = el.metadata.page_number
                category = el.category
                
                if category in ("Image", "Figure") and page_num not in pred_figs_yx:
                    pred_figs_yx.append(page_num)
                elif category == "Table" and page_num not in pred_tbls_yx:
                    pred_tbls_yx.append(page_num)
        except Exception as e:
            print(f"❌ YOLOX partition failed for {paper_id}: {e}")
            
        yolox_latency = (time.time() - start_time) * 1000  # ms
        results["yolox"]["latencies"].append(yolox_latency)
        
        yx_fig_p, yx_fig_r, yx_fig_f = calculate_metrics(gt["figures"], pred_figs_yx, num_pages)
        yx_tbl_p, yx_tbl_r, yx_tbl_f = calculate_metrics(gt["tables"], pred_tbls_yx, num_pages)
        
        results["yolox"]["figures"].append((yx_fig_p, yx_fig_r, yx_fig_f))
        results["yolox"]["tables"].append((yx_tbl_p, yx_tbl_r, yx_tbl_f))
        
        print(f"\n📄 Paper: {paper_id} ({gt['title'][:40]}...)")
        print(f"  PyMuPDF (fitz) - Latency: {pymupdf_latency:.2f}ms")
        print(f"    Figures: P={py_fig_p:.2f}, R={py_fig_r:.2f}, F1={py_fig_f:.2f} (Pred pages: {pred_figs_py})")
        print(f"    Tables:  P={py_tbl_p:.2f}, R={py_tbl_r:.2f}, F1={py_tbl_f:.2f} (Pred pages: {pred_tbls_py})")
        print(f"  YOLOX (Hi-Res) - Latency: {yolox_latency:.2f}ms")
        print(f"    Figures: P={yx_fig_p:.2f}, R={yx_fig_r:.2f}, F1={yx_fig_f:.2f} (Pred pages: {pred_figs_yx})")
        print(f"    Tables:  P={yx_tbl_p:.2f}, R={yx_tbl_r:.2f}, F1={yx_tbl_f:.2f} (Pred pages: {pred_tbls_yx})")

    return results

def run_clip_evaluation():
    print("\n" + "="*80)
    print("🚀 PHASE 2: EVALUATING CLIP VISION RETRIEVAL")
    print("="*80)
    
    # 1. Load QA pairs from evaluation_dataset.json
    dataset_path = Path("papers download/evaluation_dataset.json")
    if not dataset_path.exists():
        print(f"❌ Error: evaluation_dataset.json not found at {dataset_path}.")
        return None
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
        
    visual_queries = []
    for paper_id, data in qa_data.items():
        if paper_id not in GROUND_TRUTH:
            continue
        for qa in data.get("qa_pairs", []):
            q_type = qa.get("question_type")
            if q_type in ("figure", "table"):
                evidence = qa.get("evidence", {})
                page = evidence.get("page")
                element_label = evidence.get("figure") or evidence.get("table")
                
                visual_queries.append({
                    "paper_id": paper_id,
                    "query": qa["question"],
                    "gold_page": page,
                    "gold_label": element_label,
                    "type": q_type
                })
                
    print(f"Extracted {len(visual_queries)} visual queries (type: figure/table) from gold benchmark dataset.")
    if not visual_queries:
        print("No visual queries found in dataset.")
        return None
        
    # 2. Index YOLOX extracted images/tables in a temporary FAISS index
    print("Building temporary visual CLIP FAISS Index...")
    
    # Locate all visual files in extracted_images directory
    image_dir = Path("extracted_images")
    if not image_dir.exists():
        image_dir.mkdir(parents=True, exist_ok=True)
        
    visual_files = list(image_dir.glob("*.*"))
    # Filter for standard formats
    visual_files = [f for f in visual_files if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
    
    if not visual_files:
        print("⚠️ No visual image files found in extracted_images directory. Cannot evaluate CLIP.")
        return None
        
    print(f"Found {len(visual_files)} image files in registry. Generating CLIP embeddings...")
    
    # Load CLIP embeddings
    embeddings = embed_image_batch([str(f) for f in visual_files])
    embeddings = np.array(embeddings, dtype="float32")
    
    # Create FAISS Index
    index = faiss.IndexFlatIP(512)
    index.add(embeddings)
    
    # 3. Query the index and measure Recall & MRR
    r1_hits = 0
    r3_hits = 0
    rr_sum = 0.0
    
    for vq in visual_queries:
        q_text = vq["query"]
        gold_page = vq["gold_page"]
        paper_id = vq["paper_id"]
        
        # Embed query text
        q_vec = embed_clip_text(q_text)
        
        # Search Top 5
        scores, indices = index.search(q_vec, k=min(5, index.ntotal))
        
        # Find correct target match rank
        correct_rank = None
        for rank, idx in enumerate(indices[0], start=1):
            if idx == -1 or idx >= len(visual_files):
                continue
                
            matched_file = visual_files[idx]
            stem = matched_file.stem
            
            # Parse page number and metadata from filename
            # Example filename: 1408.5882_page_4_table_2.png
            # Or table_1408.5882_page_4_element_2.png
            page_match = re.search(r'page_(\d+)', stem, re.IGNORECASE)
            paper_match = paper_id in stem
            
            if paper_match and page_match and int(page_match.group(1)) == gold_page:
                correct_rank = rank
                break
                
        if correct_rank == 1:
            r1_hits += 1
            r3_hits += 1
            rr_sum += 1.0
        elif correct_rank in (2, 3):
            r3_hits += 1
            rr_sum += 1.0 / correct_rank
        elif correct_rank is not None:
            rr_sum += 1.0 / correct_rank
            
    recall_1 = r1_hits / len(visual_queries)
    recall_3 = r3_hits / len(visual_queries)
    mrr = rr_sum / len(visual_queries)
    
    print(f"\nCLIP Retrieval Evaluation results:")
    print(f"  Recall@1: {recall_1:.4f}")
    print(f"  Recall@3: {recall_3:.4f}")
    print(f"  MRR:      {mrr:.4f}")
    
    return {
        "queries_count": len(visual_queries),
        "recall_1": recall_1,
        "recall_3": recall_3,
        "mrr": mrr
    }

def update_report(layout_results, clip_results):
    """Updates evaluation_what_we_are_doing.md with the latest scores."""
    if not EVALUATION_REPORT_PATH.exists():
        print(f"Error: Report file not found at {EVALUATION_REPORT_PATH}")
        return
        
    with open(EVALUATION_REPORT_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Calculate average scores
    avg_py_fig_p = np.mean([r[0] for r in layout_results["pymupdf"]["figures"]])
    avg_py_fig_r = np.mean([r[1] for r in layout_results["pymupdf"]["figures"]])
    avg_py_fig_f = np.mean([r[2] for r in layout_results["pymupdf"]["figures"]])
    avg_py_tbl_p = np.mean([r[0] for r in layout_results["pymupdf"]["tables"]])
    avg_py_tbl_r = np.mean([r[1] for r in layout_results["pymupdf"]["tables"]])
    avg_py_tbl_f = np.mean([r[2] for r in layout_results["pymupdf"]["tables"]])
    avg_py_latency = np.mean(layout_results["pymupdf"]["latencies"])

    avg_yx_fig_p = np.mean([r[0] for r in layout_results["yolox"]["figures"]])
    avg_yx_fig_r = np.mean([r[1] for r in layout_results["yolox"]["figures"]])
    avg_yx_fig_f = np.mean([r[2] for r in layout_results["yolox"]["figures"]])
    avg_yx_tbl_p = np.mean([r[0] for r in layout_results["yolox"]["tables"]])
    avg_yx_tbl_r = np.mean([r[1] for r in layout_results["yolox"]["tables"]])
    avg_yx_tbl_f = np.mean([r[2] for r in layout_results["yolox"]["tables"]])
    avg_yx_latency = np.mean(layout_results["yolox"]["latencies"])

    # Update Table 1: Ingestion Results
    yolox_fig_row = f"| **YOLOX (Hi-Res)** | Figures / Images | {avg_yx_fig_p:.4f} | {avg_yx_fig_r:.4f} | {avg_yx_fig_f:.4f} | {avg_yx_latency:.2f}ms |"
    yolox_tbl_row = f"| **YOLOX (Hi-Res)** | Tables | {avg_yx_tbl_p:.4f} | {avg_yx_tbl_r:.4f} | {avg_yx_tbl_f:.4f} | {avg_yx_latency:.2f}ms |"
    pymupdf_fig_row = f"| **PyMuPDF (fitz)** | Figures / Images | {avg_py_fig_p:.4f} | {avg_py_fig_r:.4f} | {avg_py_fig_f:.4f} | {avg_py_latency:.2f}ms |"
    pymupdf_tbl_row = f"| **PyMuPDF (fitz)** | Tables | {avg_py_tbl_p:.4f} | {avg_py_tbl_r:.4f} | {avg_py_tbl_f:.4f} | {avg_py_latency:.2f}ms |"

    # Replace placeholders in Table 1
    content = content.replace("| **YOLOX (Hi-Res)** | Figures / Images | *TBD* | *TBD* | *TBD* | *TBD* |", yolox_fig_row)
    content = content.replace("| **YOLOX (Hi-Res)** | Tables | *TBD* | *TBD* | *TBD* | *TBD* |", yolox_tbl_row)
    content = content.replace("| **PyMuPDF (fitz)** | Figures / Images | *TBD* | *TBD* | *TBD* | *TBD* |", pymupdf_fig_row)
    content = content.replace("| **PyMuPDF (fitz)** | Tables | *TBD* | *TBD* | *TBD* | *TBD* |", pymupdf_tbl_row)
    content = content.replace("*Status: Pending Execution*", "*Status: Completed successfully*")

    # Update Table 2: CLIP Results
    if clip_results:
        clip_row = f"| **CLIP (ViT-B/32) FAISS** | {clip_results['queries_count']} | {clip_results['recall_1']:.4f} | {clip_results['recall_3']:.4f} | {clip_results['mrr']:.4f} |"
        content = content.replace("| **CLIP (ViT-B/32) FAISS** | *TBD* | *TBD* | *TBD* | *TBD* |", clip_row)

    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"✓ Updated evaluation tracking report at: {EVALUATION_REPORT_PATH}")

def main():
    layout_results = run_layout_evaluation()
    clip_results = run_clip_evaluation()
    update_report(layout_results, clip_results)

if __name__ == "__main__":
    main()
