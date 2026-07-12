import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.core.generator.client import generate_content_with_retry

def generate_summary(chunks):
    """
    Generates a structured, comprehensive summary of a document.
    Uses a Hybrid Router:
    - If document size <= 60,000 characters, uses Direct Single-Pass Summarization.
    - If document size > 60,000 characters, uses Hierarchical Map-Reduce (section summaries -> merge).
    """
    if not chunks:
        return "Error: No chunks provided for summarization."
        
    filename = chunks[0]["source_file"] if chunks else "Document"
    total_chars = sum(len(chunk.get("text", "")) for chunk in chunks)
    threshold = 60000 # ~12,000 tokens
    
    print(f"SYSTEM LOG: Summarization request for '{filename}' | Total size: {total_chars} chars (Threshold: {threshold})")
    
    if total_chars <= threshold:
        print("SYSTEM LOG: Document size within threshold. Executing Direct Single-Pass Summarization...")
        # Direct single-pass reconstruction
        full_text = "\n\n".join([chunk["text"] for chunk in chunks])
        
        prompt = f"""
You are a senior scientific research analyst.
Analyze the following document text extracted from the PDF '{filename}' and write a structured, comprehensive summary.

Your summary must include:
1. **Core Overview**: A brief high-level description of the paper/document's purpose.
2. **Key Findings / Contributions**: Bullet points of the main achievements, models introduced, or results.
3. **Methodology**: Explanation of the methods, architectures, or procedures described.
4. **Conclusion**: Main takeaways or future directions.

Be highly factual, precise, and professional.

=============================
  DOCUMENT TEXT
=============================
{full_text}
"""
        return generate_content_with_retry(prompt, temperature=0.2)
        
    else:
        print("SYSTEM LOG: Document size exceeds threshold. Executing Hierarchical Map-Reduce Summarization...")
        # 1. Generate section summaries
        section_summaries_text = generate_section_summaries(chunks)
        
        # 2. Merge section summaries into a final structured master summary
        merge_prompt = f"""
You are a senior scientific research analyst.
The following is a section-by-section summary of the document '{filename}'.
Merge these section summaries into a single, cohesive, structured final master summary.

Your final master summary must include:
1. **Core Overview**: A brief high-level description of the paper/document's purpose.
2. **Key Findings / Contributions**: Bullet points of the main achievements, models introduced, or results.
3. **Methodology**: Explanation of the methods, architectures, or procedures described.
4. **Conclusion**: Main takeaways or future directions.

Be highly factual, precise, and professional.

=============================
  SECTION SUMMARIES
=============================
{section_summaries_text}
"""
        print("SYSTEM LOG: Merging section summaries into master summary...")
        return generate_content_with_retry(merge_prompt, temperature=0.2)

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
