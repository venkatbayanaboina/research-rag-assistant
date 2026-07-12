import os
import sys
import json
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import config
from src.core.generator.client import generate_content_with_retry

def get_key_images_for_doc(doc_name):
    """
    Scans the images registry for the given document and selects:
    - 1 Key Architecture Diagram / Figure
    - 1 Key Performance Table
    - 1 Key Formula / Equation
    """
    registry_path = config.IMAGES_REGISTRY_PATH
    if not os.path.exists(registry_path):
        return []
        
    try:
        with open(registry_path) as f:
            registry = json.load(f)
    except:
        return []
        
    doc_base = os.path.basename(doc_name)
    doc_images = [img for img in registry if os.path.basename(img.get("source_file", "")) == doc_base]
    
    selected_images = []
    figure_keywords = ["figure", "architecture", "model", "network", "diagram", "encoder", "decoder", "flow"]
    table_keywords = ["table", "results", "bleu", "accuracy", "performance", "comparison"]
    equation_keywords = ["equation", "formula", "loss", "softmax", "sum", "="]
    
    found_fig = False
    found_tbl = False
    found_eq = False
    
    for img in doc_images:
        caption = img.get("caption", "").lower()
        cat = img.get("category", "").lower()
        path = img.get("image_path", "")
        if not os.path.exists(path):
            continue
            
        if not found_fig and (cat == "image" or any(k in caption for k in figure_keywords)):
            selected_images.append(img)
            found_fig = True
        elif not found_tbl and (cat == "table" or any(k in caption for k in table_keywords)):
            selected_images.append(img)
            found_tbl = True
        elif not found_eq and (cat == "equation" or any(k in caption for k in equation_keywords)):
            selected_images.append(img)
            found_eq = True
            
    # Fallback if no specific categories found
    if not selected_images and doc_images:
        for img in doc_images[:2]:
            path = img.get("image_path", "")
            if os.path.exists(path):
                selected_images.append(img)
                
    return selected_images

def generate_summary(chunks):
    """
    Generates a structured, comprehensive summary of a document.
    Uses a Hybrid Router:
    - If document size <= 60,000 characters, uses Direct Single-Pass Summarization.
    - If document size > 60,000 characters, uses Hierarchical Map-Reduce (section summaries -> merge).
    Attaches key figures/tables/formulas multimodally.
    """
    if not chunks:
        return "Error: No chunks provided for summarization."
        
    filename = chunks[0]["source_file"] if chunks else "Document"
    total_chars = sum(len(chunk.get("text", "")) for chunk in chunks)
    threshold = 60000 # ~12,000 tokens
    
    print(f"SYSTEM LOG: Summarization request for '{filename}' | Total size: {total_chars} chars (Threshold: {threshold})")
    
    # Load and open key visual elements
    key_images = get_key_images_for_doc(filename)
    pil_images = []
    image_info = []
    for img_item in key_images:
        try:
            pil_images.append(Image.open(img_item["image_path"]))
            image_info.append(f"- Page {img_item['page']}: {img_item.get('category', 'Element')} (Caption text: {img_item.get('caption', '')[:100]}...)")
        except:
            pass
    image_info_str = "\n".join(image_info) if image_info else "No diagrams or tables attached."
    
    if total_chars <= threshold:
        print("SYSTEM LOG: Document size within threshold. Executing Direct Single-Pass Summarization...")
        full_text = "\n\n".join([chunk["text"] for chunk in chunks])
        
        prompt = f"""
You are a senior scientific research analyst.
Analyze the following document text extracted from the PDF '{filename}' and the attached key diagrams/tables.

Your summary must include:
1. **Core Overview**: A brief high-level description of the paper/document's purpose.
2. **Key Findings / Contributions**: Bullet points of the main achievements, models introduced, or results.
3. **Methodology**: Explanation of the methods, architectures, or procedures described. Mention key formulas, equations, or architecture diagrams visible in the attached images.
4. **Conclusion**: Main takeaways or future directions.

Attached Figures/Tables details:
{image_info_str}

=============================
  DOCUMENT TEXT
=============================
{full_text}
"""
        contents = [prompt] + pil_images
        return generate_content_with_retry(contents, temperature=0.2)
        
    else:
        print("SYSTEM LOG: Document size exceeds threshold. Executing Hierarchical Map-Reduce Summarization...")
        # 1. Generate section summaries
        section_summaries_text = generate_section_summaries(chunks)
        
        # 2. Merge section summaries into a final structured master summary
        merge_prompt = f"""
You are a senior scientific research analyst.
The following is a section-by-section summary of the document '{filename}'.
Merge these section summaries and the attached diagrams/tables into a single, cohesive, structured final master summary.

Your final master summary must include:
1. **Core Overview**: A brief high-level description of the paper/document's purpose.
2. **Key Findings / Contributions**: Bullet points of the main achievements, models introduced, or results.
3. **Methodology**: Explanation of the methods, architectures, or procedures described. Mention key formulas, equations, or architecture diagrams visible in the attached images.
4. **Conclusion**: Main takeaways or future directions.

Attached Figures/Tables details:
{image_info_str}

=============================
  SECTION SUMMARIES
=============================
{section_summaries_text}
"""
        print("SYSTEM LOG: Merging section summaries into master summary...")
        contents = [merge_prompt] + pil_images
        return generate_content_with_retry(contents, temperature=0.2)

def generate_section_summaries(chunks):
    """
    Generates an executive summary of a document broken down section-by-section.
    Performs a single consolidated API call for maximum speed and cohesion.
    """
    filename = chunks[0]["source_file"] if chunks else "Document"
    
    # Group text chunks by section title (preserving document order)
    sections = {}
    section_order = []
    for chunk in chunks:
        title = chunk.get("section_title", "Introduction")
        if title not in sections:
            sections[title] = []
            section_order.append(title)
        sections[title].append(chunk["text"])
        
    # Format sections context
    sections_context = ""
    for title in section_order:
        text = "\n".join(sections[title])
        sections_context += f"\n\n--- SECTION: {title} ---\n{text}"
        
    prompt = f"""
You are a senior scientific research analyst.
Analyze the following document '{filename}', which is divided into sections.
For each section listed under 'DOCUMENT SECTIONS', write a structured, concise executive summary (2-3 sentences) detailing its core purpose, methodology, or findings.

Format your output as a markdown section-by-section breakdown:
### [Section Title]
* [Summary text...]

=============================
  DOCUMENT SECTIONS
=============================
{sections_context}
"""

    print(f"Generating section-wise summary for {filename} with Gemini...")
    return generate_content_with_retry(prompt, temperature=0.2)
