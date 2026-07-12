import os
import json
import time
import torch
import faiss
import numpy as np
from pathlib import Path
import fitz  # PyMuPDF (super fast, stable)
from PIL import Image
import io
from sentence_transformers import SentenceTransformer

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "evaluation_suite" / "pdfs"
STORAGE_DIR = PROJECT_ROOT / "storage"
IMAGES_REGISTRY_PATH = STORAGE_DIR / "images_registry.json"
IMAGE_INDEX_PATH = STORAGE_DIR / "image_db.faiss"
EXTRACTED_IMAGE_DIR = STORAGE_DIR / "extracted_images"

os.makedirs(EXTRACTED_IMAGE_DIR, exist_ok=True)

# Detect MPS (Apple Silicon GPU), CUDA, or CPU
DEVICE = "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"

print(f"🖥️ Using device: {DEVICE} for local CLIP model.")

# Load CLIP model
print("Loading CLIP model clip-ViT-B-32...")
clip_model = SentenceTransformer("clip-ViT-B-32", device=DEVICE)

# Resume check
indexed_images_files = set()
if os.path.exists(IMAGES_REGISTRY_PATH):
    try:
        with open(IMAGES_REGISTRY_PATH) as f:
            registry = json.load(f)
            indexed_images_files = {os.path.basename(img["source_file"]) for img in registry if img.get("source_file")}
    except Exception: pass

pdf_files = sorted(list(PDF_DIR.glob("*.pdf")))
papers_to_process = [p for p in pdf_files if p.name not in indexed_images_files]

print(f"📂 Total papers found locally: {len(pdf_files)}")
print(f"⏭️ Already indexed: {len(indexed_images_files)}")
print(f"⏳ Pending local processing: {len(papers_to_process)}")

if len(papers_to_process) == 0:
    print("✅ All papers are already diagram-indexed!")
else:
    # ── Extraction Loop ──────────────────────────────────────────────────────
    print("\nStarting local PyMuPDF diagram extraction...")
    for idx, pdf_path in enumerate(papers_to_process, start=1):
        t0 = time.time()
        image_chunks = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    image_list = page.get_images(full=True)
                    
                    for img_idx, img in enumerate(image_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Skip very small images (icons, logos, headers)
                        if len(image_bytes) < 5000:
                            continue
                            
                        # Save image file to storage
                        img_name = f"{pdf_path.stem}_p{page_num+1}_img{img_idx+1}.{image_ext}"
                        img_save_path = EXTRACTED_IMAGE_DIR / img_name
                        
                        with open(img_save_path, "wb") as f:
                            f.write(image_bytes)
                            
                        # Store relative path for portability
                        rel_img_path = f"storage/extracted_images/{img_name}"
                        
                        image_chunks.append({
                            "image_path": rel_img_path,
                            "page": page_num + 1,
                            "source_file": os.path.basename(pdf_path),
                            "caption": f"Extracted image {img_idx+1} on page {page_num+1}"
                        })

            # Save to FAISS and JSON index
            if image_chunks:
                image_registry = []
                if os.path.exists(IMAGES_REGISTRY_PATH):
                    try:
                        with open(IMAGES_REGISTRY_PATH) as f: image_registry = json.load(f)
                    except Exception: pass
                    
                image_index = faiss.IndexFlatIP(512)
                if os.path.exists(IMAGE_INDEX_PATH):
                    try:
                        image_index = faiss.read_index(str(IMAGE_INDEX_PATH))
                    except Exception: pass
                    
                # Load images to embed
                imgs = []
                for c in image_chunks:
                    full_path = PROJECT_ROOT / c["image_path"]
                    imgs.append(Image.open(full_path).convert("RGB"))
                    
                embeddings = clip_model.encode(imgs, show_progress_bar=False)
                image_index.add(np.array(embeddings, dtype="float32"))
                
                start_id = len(image_registry)
                for i, chunk in enumerate(image_chunks):
                    chunk["image_id"] = start_id + i
                    image_registry.append(chunk)
                    
                with open(IMAGES_REGISTRY_PATH, "w") as f: json.dump(image_registry, f, indent=2)
                faiss.write_index(image_index, str(IMAGE_INDEX_PATH))
                
                print(f"[{idx}/{len(papers_to_process)}] ✓ {pdf_path.name}: {len(image_chunks)} diagrams in {round(time.time() - t0, 2)}s")
            else:
                # Mark processed to skip in future resume
                image_registry = []
                if os.path.exists(IMAGES_REGISTRY_PATH):
                    try:
                        with open(IMAGES_REGISTRY_PATH) as f: image_registry = json.load(f)
                    except Exception: pass
                
                dummy_record = {
                    "image_path": "",
                    "page": 0,
                    "source_file": os.path.basename(pdf_path),
                    "caption": "No images found"
                }
                image_registry.append(dummy_record)
                with open(IMAGES_REGISTRY_PATH, "w") as f: json.dump(image_registry, f, indent=2)
                
                print(f"[{idx}/{len(papers_to_process)}] ✓ {pdf_path.name}: 0 diagrams in {round(time.time() - t0, 2)}s")
                
        except Exception as e:
            print(f"[{idx}/{len(papers_to_process)}] ❌ Failed: {pdf_path.name}: {e}")

    print("\n🎉 Local PyMuPDF Diagram Ingestion Complete!")
