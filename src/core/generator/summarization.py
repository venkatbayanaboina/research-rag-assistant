import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.core.generator.client import generate_content_with_retry

def generate_summary(chunks):
    """
    Generates a structured, comprehensive summary of a document
    using the text extracted from its chunks.
    """
    # Reconstruct document text
    full_text = "\n\n".join([chunk["text"] for chunk in chunks])
    filename = chunks[0]["source_file"] if chunks else "Document"
    
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

    print(f"Generating summary for {filename} with Gemini...")
    return generate_content_with_retry(prompt, temperature=0.2)

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
