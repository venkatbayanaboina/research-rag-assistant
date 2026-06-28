import os
from unstructured.partition.pdf import partition_pdf

def parse_pdf(file_path, strategy="fast"):
    """
    Parses a PDF document using the unstructured library.
    Supports 'fast' (text extraction) and 'hi_res' (layout parsing models).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")

    kwargs = {
        "filename": file_path,
        "strategy": strategy,
    }
    
    if strategy == "hi_res":
        kwargs.update({
            "infer_table_structure": True,
            "extract_image_block_types": ["Image"],
            "extract_image_block_to_payload": True
        })
        
    print(f"Parsing PDF with unstructured (strategy: {strategy}): {os.path.basename(file_path)}...")
    elements = partition_pdf(**kwargs)
    print(f"Parsed {len(elements)} elements from {os.path.basename(file_path)}.")
    return elements
