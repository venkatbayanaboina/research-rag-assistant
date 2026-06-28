import os
# Prevent Loky multiprocessing segmentation faults on Apple Silicon macOS
os.environ["UNSTRUCTURED_PARALLEL"] = "False"

import sys
from unstructured.partition.pdf import partition_pdf

# Include root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

def parse_pdf(file_path, strategy="fast"):
    """
    Parses a PDF document using the unstructured library.
    Supports 'fast' (text-only) and 'hi_res' (layout parsing & image extraction).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")

    kwargs = {
        "filename": file_path,
        "strategy": strategy,
    }
    
    if strategy == "hi_res":
        # Extract visual structures as standalone PNG files
        kwargs.update({
            "extract_image_block_types": ["Image", "Table"],
            "extract_image_block_output_dir": config.EXTRACTED_IMAGE_DIR,
            "extract_image_block_to_payload": False,
            "hi_res_model_name": "yolox"
        })
        
    print(f"Parsing PDF with unstructured (strategy: {strategy}): {os.path.basename(file_path)}...")
    elements = partition_pdf(**kwargs)
    print(f"Parsed {len(elements)} elements from {os.path.basename(file_path)}.")
    return elements
