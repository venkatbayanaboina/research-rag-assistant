import os
# Prevent Loky multiprocessing segmentation faults on Apple Silicon macOS
os.environ["UNSTRUCTURED_PARALLEL"] = "False"

import sys
from unstructured.partition.pdf import partition_pdf
from pdf2image import pdfinfo_from_path

# Include root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

def get_pdf_page_count(file_path):
    """Returns the total number of pages in the PDF file using pdfinfo."""
    try:
        info = pdfinfo_from_path(file_path)
        return int(info.get("Pages", 1))
    except Exception as e:
        print(f"Failed to get page count using pdfinfo: {e}")
        return 1

def parse_pdf_progressive(file_path, strategy="fast", progress_callback=None):
    """
    Parses a PDF page-by-page to allow progress tracking and avoid memory spikes.
    Calls progress_callback(current_page, total_pages) if provided.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
        
    total_pages = get_pdf_page_count(file_path)
    print(f"Starting progressive parsing of {os.path.basename(file_path)} ({total_pages} pages)...")
    
    all_elements = []
    
    kwargs = {
        "filename": file_path,
        "strategy": strategy,
    }
    
    if strategy == "hi_res":
        kwargs.update({
            "extract_image_block_types": ["Image", "Table"],
            "extract_image_block_output_dir": config.EXTRACTED_IMAGE_DIR,
            "extract_image_block_to_payload": False,
            "hi_res_model_name": "yolox"
        })
        
    for page_num in range(1, total_pages + 1):
        if progress_callback:
            progress_callback(page_num, total_pages)
        else:
            print(f"Parsing page {page_num}/{total_pages}...")
            
        try:
            # Parse only the current page (1-indexed)
            page_kwargs = kwargs.copy()
            page_kwargs["page_numbers"] = [page_num]
            elements = partition_pdf(**page_kwargs)
            all_elements.extend(elements)
        except Exception as e:
            print(f"Error parsing page {page_num}: {e}. Continuing...")
            
    print(f"Finished progressive parsing. Extracted {len(all_elements)} elements.")
    return all_elements
