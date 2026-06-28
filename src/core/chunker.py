import os
import sys
from unstructured.chunking.title import chunk_by_title

# Include root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

def process_text_chunks(elements, source_filename):
    """
    Groups unstructured elements using chunk_by_title and maps them into dictionaries.
    These will be indexed using BGE text embeddings.
    """
    print("Formatting text chunks...")
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
        
        # Keep table/image metadata reference for visual representation if present
        if hasattr(chunk.metadata, "orig_elements"):
            for element in chunk.metadata.orig_elements:
                if element.category == "Image":
                    chunk_dict["images"].append({
                        "page": getattr(element.metadata, "page_number", chunk_dict["page"]),
                        "ocr_text": element.text,
                    })
                elif element.category == "Table":
                    chunk_dict["tables"].append({
                        "page": getattr(element.metadata, "page_number", chunk_dict["page"]),
                        "text": element.text,
                        "html": getattr(element.metadata, "text_as_html", "")
                    })
                    
        processed_chunks.append(chunk_dict)
        
    print(f"Generated {len(processed_chunks)} text chunks for {os.path.basename(source_filename)}.")
    return processed_chunks

def process_image_chunks(elements, source_filename):
    """
    Extracts elements representing visual diagrams or tables that have cropped image files.
    These will be indexed using CLIP image embeddings.
    """
    print("Formatting visual image chunks...")
    image_chunks = []
    for el in elements:
        if el.category in ["Image", "Table"]:
            image_path = getattr(el.metadata, "image_path", None)
            if image_path and os.path.exists(image_path):
                # Standardize paths to use forward slashes
                normalized_path = os.path.abspath(image_path).replace("\\", "/")
                image_chunks.append({
                    "image_id": None,  # Dynamically set when added to vector store
                    "source_file": os.path.basename(source_filename),
                    "page": getattr(el.metadata, "page_number", 1),
                    "category": el.category,
                    "image_path": normalized_path,
                    "caption": el.text.strip()
                })
                
    print(f"Generated {len(image_chunks)} visual chunks for {os.path.basename(source_filename)}.")
    return image_chunks
