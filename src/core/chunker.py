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
    import time
    from src.core.utils.profiler import log_timing
    
    start_time = time.time()
    print("Formatting text chunks...")
    chunks = chunk_by_title(
        elements,
        max_characters=config.CHUNK_MAX_CHARACTERS,
        new_after_n_chars=config.CHUNK_NEW_AFTER_CHARACTERS,
        combine_text_under_n_chars=config.CHUNK_COMBINE_UNDER_CHARACTERS,
        overlap=config.CHUNK_OVERLAP
    )
    
    processed_chunks = []
    current_section = "Introduction"
    
    for chunk in chunks:
        # Scan constituent elements to identify if a new section starts in this chunk
        if hasattr(chunk.metadata, "orig_elements"):
            for element in chunk.metadata.orig_elements:
                if element.category in ["Title", "SectionHeader"] or (element.category == "Header" and len(element.text.strip()) < 100):
                    # Clean section header name (remove redundant spaces/newlines)
                    header_text = " ".join(element.text.strip().split())
                    if header_text:
                        current_section = header_text
                        break
                        
        chunk_dict = {
            "chunk_id": None,  # Dynamically set when added to main store
            "source_file": os.path.basename(source_filename),
            "text": chunk.text,
            "page": getattr(chunk.metadata, "page_number", 1),
            "section_title": current_section,
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
        
    duration = time.time() - start_time
    log_timing(
        step_name="text_chunking",
        duration_seconds=duration,
        metadata={
            "file_name": os.path.basename(source_filename),
            "chunk_count": len(processed_chunks)
        }
    )
    return processed_chunks

def process_image_chunks(elements, source_filename):
    """
    Extracts elements representing visual diagrams or tables that have cropped image files.
    Associates nearby figure/table captions with the visual chunks for robust retrieval.
    These will be indexed using CLIP image embeddings.
    """
    import time
    from src.core.utils.profiler import log_timing
    
    start_time = time.time()
    print("Formatting visual image chunks...")
    image_chunks = []
    
    for idx, el in enumerate(elements):
        if el.category in ["Image", "Table"]:
            image_path = getattr(el.metadata, "image_path", None)
            if image_path and os.path.exists(image_path):
                # Standardize paths to use forward slashes
                normalized_path = os.path.abspath(image_path).replace("\\", "/")
                el_page = getattr(el.metadata, "page_number", 1)
                
                # Heuristics-based caption association within +/- 8 elements window
                candidates = []
                for offset in range(-8, 9):
                    if offset == 0:
                        continue
                    neighbor_idx = idx + offset
                    if 0 <= neighbor_idx < len(elements):
                        neighbor = elements[neighbor_idx]
                        neighbor_page = getattr(neighbor.metadata, "page_number", None)
                        
                        # Only associate if on the same page
                        if neighbor_page == el_page:
                            text = neighbor.text.strip()
                            text_lower = text.lower()
                            
                            is_caption_cat = neighbor.category in ["FigureCaption", "TableCaption", "Caption"]
                            is_caption_text = text_lower.startswith(("fig", "figure", "table", "chart", "diagram"))
                            
                            if is_caption_cat or is_caption_text:
                                # Determine if this neighbor is a semantic type match
                                is_table_type = text_lower.startswith(("table", "tab."))
                                is_fig_type = text_lower.startswith(("fig", "figure"))
                                
                                type_match = False
                                if el.category == "Table" and is_table_type:
                                    type_match = True
                                elif el.category == "Image" and is_fig_type:
                                    type_match = True
                                    
                                # Determine if neighbor is in the expected relative position
                                # Figures usually have captions below them (offset > 0)
                                # Tables usually have captions above them (offset < 0)
                                correct_rel_pos = False
                                if el.category == "Table" and offset < 0:
                                    correct_rel_pos = True
                                elif el.category == "Image" and offset > 0:
                                    correct_rel_pos = True
                                    
                                candidates.append({
                                    "text": text,
                                    "type_match": type_match,
                                    "correct_rel_pos": correct_rel_pos,
                                    "distance": abs(offset)
                                })
                
                # Sort candidates:
                # 1. distance (smaller first)
                # 2. type_match (True first)
                # 3. correct_rel_pos (True first)
                best_caption = ""
                if candidates:
                    candidates.sort(key=lambda x: (x["distance"], not x["type_match"], not x["correct_rel_pos"]))
                    best_caption = candidates[0]["text"]
                    
                ocr_text = el.text.strip()
                if best_caption:
                    # Combine the associated caption with the internal OCR text of the image box if available
                    if ocr_text and ocr_text.lower() not in best_caption.lower():
                        full_caption = f"{best_caption} | Internal Text: {ocr_text}"
                    else:
                        full_caption = best_caption
                else:
                    full_caption = ocr_text
                    
                image_chunks.append({
                    "image_id": None,  # Dynamically set when added to vector store
                    "source_file": os.path.basename(source_filename),
                    "page": el_page,
                    "category": el.category,
                    "image_path": normalized_path,
                    "caption": full_caption
                })
                
    duration = time.time() - start_time
    log_timing(
        step_name="visual_chunking",
        duration_seconds=duration,
        metadata={
            "file_name": os.path.basename(source_filename),
            "visual_chunk_count": len(image_chunks)
        }
    )
    return image_chunks
