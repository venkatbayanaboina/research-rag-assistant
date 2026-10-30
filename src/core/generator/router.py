import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# The full intent catalog. The first four are the classic chat flows; the
# research intents (ADR-008) drive multi-step agentic research actions.
RESEARCH_INTENTS = {"EVIDENCE_TABLE", "CONTRADICTION", "GAP_EXPLORATION"}
SPECIAL_INTENTS = {"SUMMARY", "SECTION_SUMMARY", "COMPARISON"} | RESEARCH_INTENTS
ALL_INTENTS = {"STANDARD_CHAT"} | SPECIAL_INTENTS


def route_user_intent(prompt, indexed_files):
    """
    Classifies the user's request into one of ALL_INTENTS and resolves which
    indexed files they refer to. For research intents it also extracts the
    research focus (the topic/question to analyse).

    Returns:
        dict: {"intent": str, "target_docs": list, "focus": str}
    """
    default_response = {"intent": "STANDARD_CHAT", "target_docs": [], "focus": ""}

    if not indexed_files:
        return default_response

    files_list_str = "\n".join([f"- {filename}" for filename in indexed_files])

    system_instruction = f"""
You are an intelligent RAG agent query router. Your job is to classify the user's request, resolve which indexed files they refer to, and extract the research focus when applicable.

The current indexed files in the database (in order of upload, oldest to newest):
{files_list_str}

Classify the request into exactly one of these intents:
1. "STANDARD_CHAT": General questions, Q&A queries (e.g. "what is self attention?"), figure/table requests, or casual remarks.
2. "SUMMARY": Executive summary of one or more documents (e.g. "summarize attention-is-all-you-need-Paper.pdf").
3. "SECTION_SUMMARY": Section-wise or chapter breakdowns of a document.
4. "COMPARISON": Compare, contrast, or list differences between two or more documents.
5. "EVIDENCE_TABLE": Build a structured evidence table extracting per-paper fields such as method, dataset, metric, reported result, limitations (e.g. "build an evidence table for these papers", "what method, dataset and results does each paper report?").
6. "CONTRADICTION": Identify whether papers disagree or conflict on a specific topic (e.g. "do my papers disagree about X?", "contradictions between the papers on Y").
7. "GAP_EXPLORATION": Explore underexplored areas or open problems across the indexed papers (e.g. "what are the research gaps in my papers?", "what has not been studied yet?").

For intents 5-7, also extract "focus": the specific topic, method, or question the user wants analysed (use their phrasing; may be empty if they mean the whole papers).

CRITICAL RULES:
- Queries asking to retrieve, show, describe, or explain specific figures, tables, or equations (e.g. "table 3", "fig 2") MUST be "STANDARD_CHAT".
- Only assign a research intent (5-7) when the user clearly asks for cross-paper synthesis, not a simple question about one paper.

Resolve target documents from the prompt:
- Mentioned filenames or name fragments (e.g. "attention paper") → add that file.
- "the last paper(s)" → the most recent item(s) in the list. "all papers" → all items.
- If no document is identifiable, leave target_docs empty.

Return your decision strictly in JSON format matching this schema:
{{
  "intent": "STANDARD_CHAT" | "SUMMARY" | "SECTION_SUMMARY" | "COMPARISON" | "EVIDENCE_TABLE" | "CONTRADICTION" | "GAP_EXPLORATION",
  "target_docs": [list of matched filenames],
  "focus": "topic or question for research intents, else empty string"
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

        # Schema validation: unknown intents and unresolvable targets fall back.
        if result.get("intent") in ALL_INTENTS:
            result["target_docs"] = [f for f in result.get("target_docs", []) if f in indexed_files]
            if result["intent"] in SPECIAL_INTENTS and not result["target_docs"]:
                result["intent"] = "STANDARD_CHAT"
            focus = result.get("focus")
            result["focus"] = focus.strip() if isinstance(focus, str) else ""
            return result
    except Exception as e:
        print(f"LLM Intent routing failed: {e}")

    return default_response
