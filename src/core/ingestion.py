import os
# Prevent Loky multiprocessing segmentation faults on Apple Silicon macOS
os.environ["UNSTRUCTURED_PARALLEL"] = "False"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import sys
from unstructured.partition.pdf import partition_pdf

# Include root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

import pypdf
from unstructured.documents.elements import Title, NarrativeText, ElementMetadata

def parse_pdf(file_path, strategy="fast"):
    """
    Parses a PDF document in a single optimized call.
    If strategy is fast, uses lightweight pypdf to extract pages in milliseconds.
    If strategy is hi_res, uses unstructured partition_pdf with YOLOX.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
        
    import time
    from src.core.utils.profiler import log_timing
    start_time = time.time()
    
    if strategy == "fast":
        print(f"Parsing PDF with fast pypdf engine: {os.path.basename(file_path)}...")
        elements = []
        try:
            reader = pypdf.PdfReader(file_path)
            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                text = page.extract_text() or ""
                # Segment page text into clean paragraphs
                paragraphs = text.split("\n\n")
                for para in paragraphs:
                    para_clean = para.strip()
                    if para_clean:
                        # Clean title detection logic
                        is_title = len(para_clean) < 150 and not para_clean.endswith(".")
                        meta = ElementMetadata(page_number=page_num)
                        if is_title:
                            elements.append(Title(text=para_clean, metadata=meta))
                        else:
                            elements.append(NarrativeText(text=para_clean, metadata=meta))
        except Exception as e:
            print(f"pypdf extraction failed, falling back to basic extraction: {e}")
            
        duration = time.time() - start_time
        log_timing(
            step_name="layout_selection_and_parsing",
            duration_seconds=duration,
            metadata={
                "file_name": os.path.basename(file_path),
                "strategy": strategy,
                "element_count": len(elements)
            }
        )
        return elements
        
    # strategy == "hi_res"
    kwargs = {
        "filename": file_path,
        "strategy": strategy,
    }
    
    if strategy == "hi_res":
        from src.core.vector_store import get_paths
        paths = get_paths()
        # Extract visual structures as standalone PNG files
        kwargs.update({
            "extract_image_block_types": ["Image", "Table"],
            "extract_image_block_output_dir": paths["extracted_images"],
            "extract_image_block_to_payload": False,
            "hi_res_model_name": "yolox"
        })
        
    print(f"Parsing PDF in one optimized call (strategy: {strategy}): {os.path.basename(file_path)}...")
    
    elements = partition_pdf(**kwargs)
    duration = time.time() - start_time
    
    log_timing(
        step_name="layout_selection_and_parsing",
        duration_seconds=duration,
        metadata={
            "file_name": os.path.basename(file_path),
            "strategy": strategy,
            "element_count": len(elements)
        }
    )
    return elements





