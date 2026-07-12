import os
import json
import sys
from google.genai import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import config

def route_user_intent(prompt, indexed_files):
    """
    Routes user request to: STANDARD_CHAT, SUMMARY, SECTION_SUMMARY, or COMPARISON
    and resolves document references based on the provided list of indexed files.
    Returns:
        dict: {"intent": str, "target_docs": list}
    """
    default_response = {"intent": "STANDARD_CHAT", "target_docs": []}
    
    if not indexed_files:
        return default_response
        
    files_list_str = "\n".join([f"- {filename}" for filename in indexed_files])
    
    system_instruction = f"""
You are an intelligent RAG agent query router. Your job is to classify the user's intent and resolve which indexed files they are referring to.

The current indexed files in the database (in order of upload, oldest to newest):
{files_list_str}

Analyze the user's query and classify it into one of these intents:
1. "STANDARD_CHAT": General questions, Q&A queries (e.g. "what is self attention?"), or when no summaries/comparisons are requested.
2. "SUMMARY": Requests for an executive summary of one or more documents (e.g. "summarize attention-is-all-you-need-Paper.pdf", "give me a summary of the papers").
3. "SECTION_SUMMARY": Requests for section-wise, detailed, or chapter breakdowns of a document (e.g. "detailed section summary of the attention paper", "section wise breakdown of the last paper").
4. "COMPARISON": Requests to compare, contrast, or list differences between two or more documents (e.g. "compare attention-is-all-you-need-Paper.pdf and Social-Engineering-Attack.pdf", "how does the transformer differ from the social engineering paper?").

Resolve target documents based on the user's prompt:
- If they mention a filename or name without extension (e.g. "attention paper" matches "attention-is-all-you-need-Paper.pdf"), add it to target_docs.
- If they say "the last paper" or "last paper", resolve it to the most recently uploaded paper (the last item in the list).
- If they say "the last two papers", resolve it to the final two items in the list.
- If they say "all papers", resolve it to all items in the list.
- If the intent is STANDARD_CHAT, keep target_docs empty unless they explicitly ask a question targeted at a specific paper (e.g., "what is the architecture of the attention paper?" -> intent is STANDARD_CHAT, but target_docs can have "attention-is-all-you-need-Paper.pdf" to narrow down search context).

Return your decision strictly in JSON format matching this schema:
{{
  "intent": "STANDARD_CHAT" | "SUMMARY" | "SECTION_SUMMARY" | "COMPARISON",
  "target_docs": [list of matched filenames]
}}
"""

    try:
        from src.core.generator.client import generate_content_with_retry
        response_text = generate_content_with_retry(
            prompt=prompt,
            temperature=0.0,
            system_instruction=system_instruction,
            response_mime_type="application/json"
        )
        result = json.loads(response_text)
        
        # Validate structure
        if "intent" in result and "target_docs" in result:
            # Clean resolved targets (ensure they actually exist in indexed_files)
            result["target_docs"] = [f for f in result["target_docs"] if f in indexed_files]
            return result
    except Exception as e:
        print(f"LLM Intent routing failed: {e}")
            
    return default_response
