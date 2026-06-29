import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import config
from src.core.vector_store import search_store, search_image_store, get_registry
from src.core.generator import generate_answer, generate_summary, generate_section_summaries, generate_comparison
from src.core.generator.router import route_user_intent

def execute_rag_pipeline(prompt, indexed_docs, use_rerank=None, chat_history=None):
    """
    Core RAG Orchestration Pipeline.
    Classifies user intent, retrieves appropriate context, and calls corresponding generation engines.
    Is completely independent of Streamlit or CLI specific rendering blocks.
    
    Returns:
        dict: {
            "intent": str,
            "answer": str,
            "image_results": list,
            "search_results": list,
            "is_special_intent": bool
        }
    """
    if use_rerank is None:
        use_rerank = config.RERANK_ENABLED
    if chat_history is None:
        chat_history = []

    # 1. Route based on detected LLM intent
    routing = route_user_intent(prompt, indexed_docs)
    intent = routing.get("intent", "STANDARD_CHAT")
    targets = routing.get("target_docs", [])

    answer = ""
    image_results = []
    search_results = []
    is_special_intent = False

    if intent == "COMPARISON" and len(targets) >= 2:
        doc_a, doc_b = targets[0], targets[1]
        try:
            registry = get_registry()
            doc_a_chunks = [chunk for chunk in registry if chunk["source_file"] == doc_a]
            doc_b_chunks = [chunk for chunk in registry if chunk["source_file"] == doc_b]
            
            if doc_a_chunks and doc_b_chunks:
                answer = generate_comparison(doc_a, doc_a_chunks, doc_b, doc_b_chunks)
            else:
                answer = f"Error: Could not retrieve chunks for both '{doc_a}' and '{doc_b}'."
        except Exception as e:
            answer = f"Error generating comparison: {e}"
        is_special_intent = True

    elif intent == "SECTION_SUMMARY" and len(targets) >= 1:
        target_doc = targets[0]
        try:
            registry = get_registry()
            doc_chunks = [chunk for chunk in registry if chunk["source_file"] == target_doc]
            
            if doc_chunks:
                answer = generate_section_summaries(doc_chunks)
            else:
                answer = f"Error: No text chunks found in registry for document '{target_doc}'."
        except Exception as e:
            answer = f"Error generating summary: {e}"
        is_special_intent = True

    elif intent == "SUMMARY" and len(targets) >= 1:
        summaries = []
        for target_doc in targets:
            try:
                registry = get_registry()
                doc_chunks = [chunk for chunk in registry if chunk["source_file"] == target_doc]
                
                if doc_chunks:
                    sum_txt = generate_summary(doc_chunks)
                    summaries.append(f"### Summary of {target_doc}\n{sum_txt}")
                else:
                    summaries.append(f"Error: No text chunks found in registry for document '{target_doc}'.")
            except Exception as e:
                summaries.append(f"Error generating summary for '{target_doc}': {e}")
        answer = "\n\n---\n\n".join(summaries)
        is_special_intent = True

    else:
        # Standard RAG Q&A
        search_results = search_store(prompt, rerank=use_rerank)
        if targets:
            # Filter results to specified targets
            search_results = [res for res in search_results if res["chunk"]["source_file"] in targets]
            
        image_results = search_image_store(prompt)
        if targets:
            # Filter visual results to specified targets
            image_results = [res for res in image_results if res["chunk"]["source_file"] in targets]
            
        try:
            answer = generate_answer(prompt, search_results, image_results, chat_history)
        except Exception as e:
            answer = f"Error generating answer: {e}"

    return {
        "intent": intent,
        "answer": answer,
        "image_results": image_results,
        "search_results": search_results,
        "is_special_intent": is_special_intent
    }
