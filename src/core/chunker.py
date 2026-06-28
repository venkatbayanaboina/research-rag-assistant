import os
import sys
from unstructured.chunking.title import chunk_by_title

# Include root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

def process_chunks(elements, source_filename):
    """
    Groups unstructured elements using chunk_by_title and maps them into dictionaries
    with source metadata.
    """
    print("Chunking document elements...")
    chunks = chunk_by_title(
        elements,
        max_characters=config.CHUNK_MAX_CHARACTERS,
        new_after_n_chars=config.CHUNK_NEW_AFTER_CHARACTERS,
        combine_text_under_n_chars=config.CHUNK_COMBINE_UNDER_CHARACTERS,
        overlap=config.CHUNK_OVERLAP
    )
    
    processed_chunks = []
    for chunk in chunks:
        chunk_dict = {
            "chunk_id": None,  # Dynamically set when added to main store
            "source_file": os.path.basename(source_filename),
            "text": chunk.text,
            "page": getattr(chunk.metadata, "page_number", 1),
            "images": [],
            "tables": []
        }
        
        if hasattr(chunk.metadata, "orig_elements"):
            for element in chunk.metadata.orig_elements:
                if element.category == "Image":
                    chunk_dict["images"].append({
                        "page": getattr(element.metadata, "page_number", chunk_dict["page"]),
                        "ocr_text": element.text,
                        "image_base64": getattr(element.metadata, "image_base64", "")
                    })
                elif element.category == "Table":
                    chunk_dict["tables"].append({
                        "page": getattr(element.metadata, "page_number", chunk_dict["page"]),
                        "text": element.text,
                        "html": getattr(element.metadata, "text_as_html", "")
                    })
                    
        processed_chunks.append(chunk_dict)
        
    print(f"Generated {len(processed_chunks)} chunks for {os.path.basename(source_filename)}.")
    return processed_chunks
