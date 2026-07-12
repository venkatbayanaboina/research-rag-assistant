import os
import sys
import json
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import config
from src.core.generator.client import generate_content_with_retry
from src.core.generator.summarization import get_key_images_for_doc

def generate_comparison(doc_a_name, doc_a_chunks, doc_b_name, doc_b_chunks):
    """
    Generates a structured side-by-side comparison of two documents.
    Outputs a clean Markdown table comparing methodology, contributions, and limits.
    Attaches key figures/tables/formulas of both documents multimodally.
    """
    doc_a_text = "\n\n".join([chunk["text"] for chunk in doc_a_chunks])
    doc_b_text = "\n\n".join([chunk["text"] for chunk in doc_b_chunks])
    
    # Retrieve key visual elements for both papers
    key_images_a = get_key_images_for_doc(doc_a_name)
    key_images_b = get_key_images_for_doc(doc_b_name)
    
    pil_images = []
    image_info = []
    
    for img_item in key_images_a:
        try:
            pil_images.append(Image.open(img_item["image_path"]))
            image_info.append(f"- Paper A (Page {img_item['page']}): {img_item.get('category', 'Visual')} - Caption: {img_item.get('caption', '')[:100]}...")
        except: pass
        
    for img_item in key_images_b:
        try:
            pil_images.append(Image.open(img_item["image_path"]))
            image_info.append(f"- Paper B (Page {img_item['page']}): {img_item.get('category', 'Visual')} - Caption: {img_item.get('caption', '')[:100]}...")
        except: pass
        
    image_info_str = "\n".join(image_info) if image_info else "No diagrams or tables attached."
    
    prompt = f"""
You are an expert scientific analyst.
Compare and contrast the following two research papers/documents:
1. Paper A: {doc_a_name}
2. Paper B: {doc_b_name}

Analyze the methodology, main achievements, performance, limitations, core architecture, and formulas of both papers, utilizing the attached diagrams/tables.

Format your answer as a structured Markdown table comparing key dimensions:
| Dimension | {doc_a_name} | {doc_b_name} |
| --- | --- | --- |
| Core Architecture (refer to diagrams) | ... | ... |
| Key Formulas & Mathematical approach | ... | ... |
| Main Contributions | ... | ... |
| Key Performance / Results (refer to tables) | ... | ... |
| Limitations | ... | ... |

Provide a brief 3-sentence summary of differences at the end.

Attached Figures/Tables details:
{image_info_str}

=============================
  CONTEXT FOR {doc_a_name}
=============================
{doc_a_text}

=============================
  CONTEXT FOR {doc_b_name}
=============================
{doc_b_text}
"""

    print(f"Generating multimodal comparison matrix between {doc_a_name} and {doc_b_name} with Gemini...")
    contents = [prompt] + pil_images
    return generate_content_with_retry(contents, temperature=0.0)
