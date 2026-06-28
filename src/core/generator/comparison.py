import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.core.generator.client import generate_content_with_retry

def generate_comparison(doc_a_name, doc_a_chunks, doc_b_name, doc_b_chunks):
    """
    Generates a structured side-by-side comparison of two documents.
    Outputs a clean Markdown table comparing methodology, contributions, and limits.
    """
    doc_a_text = "\n\n".join([chunk["text"] for chunk in doc_a_chunks])
    doc_b_text = "\n\n".join([chunk["text"] for chunk in doc_b_chunks])
    
    prompt = f"""
You are an expert scientific analyst.
Compare and contrast the following two research papers/documents:
1. Paper A: {doc_a_name}
2. Paper B: {doc_b_name}

Analyze the methodology, main achievements, performance, limitations, and core architecture of both papers.

Format your answer as a structured Markdown table comparing key dimensions, followed by a brief summary of the differences:
| Dimension | {doc_a_name} | {doc_b_name} |
| --- | --- | --- |
| Core Architecture | ... | ... |
| Main Contributions | ... | ... |
| Key Performance / Results | ... | ... |
| Limitations | ... | ... |

=============================
  CONTEXT FOR {doc_a_name}
=============================
{doc_a_text}

=============================
  CONTEXT FOR {doc_b_name}
=============================
{doc_b_text}
"""

    print(f"Generating comparison matrix between {doc_a_name} and {doc_b_name} with Gemini...")
    return generate_content_with_retry(prompt, temperature=0.2)
