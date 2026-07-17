import os
import sys
import json
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import config
from src.core.generator.client import generate_content_with_retry
from src.core.generator.summarization import get_key_images_for_doc

def generate_comparison(docs_data):
    """
    Generates a structured side-by-side comparison of multiple documents (2 to 5+).
    Outputs a clean Markdown table comparing methodology, contributions, and limits.
    docs_data: List of dicts, each with keys 'name' and 'chunks'.
    """
    pil_images = []
    image_info = []
    
    # Process key images for each document
    for doc_idx, doc in enumerate(docs_data):
        doc_name = doc["name"]
        key_images = get_key_images_for_doc(doc_name)
        for img_item in key_images:
            try:
                # Limit total attached images to prevent prompt cluttering (e.g. max 15)
                if len(pil_images) < 15:
                    pil_images.append(Image.open(img_item["image_path"]))
                    image_info.append(f"- Paper {chr(65 + doc_idx)} ({doc_name}, Page {img_item['page']}): {img_item.get('category', 'Visual')} - Caption: {img_item.get('caption', '')[:100]}...")
            except: pass
            
    image_info_str = "\n".join(image_info) if image_info else "No diagrams or tables attached."
    
    # Build prompt instructions dynamically
    paper_list_str = "\n".join([f"{idx+1}. Paper {chr(65 + idx)}: {doc['name']}" for idx, doc in enumerate(docs_data)])
    
    header_cols = ["Dimension"] + [doc["name"] for doc in docs_data]
    header_row = "| " + " | ".join(header_cols) + " |"
    separator_row = "| " + " | ".join(["---"] * len(header_cols)) + " |"
    
    example_cols = ["..."] * len(docs_data)
    example_row = "| Core Architecture (refer to diagrams) | " + " | ".join(example_cols) + " |"
    
    prompt = f"""
You are an expert scientific analyst.
Compare and contrast the following research papers/documents side-by-side:
{paper_list_str}

Analyze the methodology, main achievements, performance, limitations, core architecture, and formulas of all papers, utilizing the attached diagrams/tables.

Format your answer as a structured Markdown table comparing key dimensions. You must have one column per paper:
{header_row}
{separator_row}
{example_row}
| Key Formulas & Mathematical approach | {" | ".join(example_cols)} |
| Main Contributions | {" | ".join(example_cols)} |
| Key Performance / Results (refer to tables) | {" | ".join(example_cols)} |
| Limitations | {" | ".join(example_cols)} |

Provide a brief paragraph summarizing the key differences, relationships, and trade-offs among all compared papers at the end.

Attached Figures/Tables details:
{image_info_str}
"""

    # Append the context of each paper dynamically
    for idx, doc in enumerate(docs_data):
        doc_text = "\n\n".join([chunk["text"] for chunk in doc["chunks"]])
        prompt += f"\n\n=============================\n  CONTEXT FOR {doc['name']} (Paper {chr(65 + idx)})\n=============================\n{doc_text}\n"

    print(f"Generating multimodal comparison matrix between {[d['name'] for d in docs_data]} with Gemini...")
    contents = [prompt] + pil_images
    return generate_content_with_retry(contents, temperature=0.0)
