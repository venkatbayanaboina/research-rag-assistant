import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import config
from src.core.vector_store import search_store, search_image_store, get_registry
# Submodule imports (not the package namespace): keeps src.core.generator's
# lazy __init__ free of import cycles with this module.
from src.core.generator.answering import generate_answer
from src.core.generator.summarization import generate_summary, generate_section_summaries
from src.core.generator.comparison import generate_comparison
from src.core.generator.router import route_user_intent

def generate_similar_queries(prompt):
    """
    Asks the LLM to generate exactly 5 alternative/similar search queries.
    """
    import json
    from src.core.generator.client import generate_content_with_retry

    system_instruction = """
You are a helpful search assistant. Your task is to generate exactly 5 alternative/similar formulations of the user's search query to help retrieve matching text chunks and diagrams from a research paper database.
The alternative queries should cover synonyms, different phrasings, and query variations (including both technical/matter focus and potential layout/figure references).

Return your decision strictly in JSON format matching this schema:
{
  "queries": [
    "alternative query 1",
    "alternative query 2",
    "alternative query 3",
    "alternative query 4",
    "alternative query 5"
  ]
}
"""
    try:
        response_text = generate_content_with_retry(
            prompt=prompt,
            temperature=0.7,
            system_instruction=system_instruction,
            response_mime_type="application/json"
        )
        result = json.loads(response_text)
        if "queries" in result and isinstance(result["queries"], list):
            expanded = [q.strip() for q in result["queries"] if q.strip()]
            return expanded[:5]
    except Exception as e:
        print(f"Error generating expanded queries: {e}")
    return []


def execute_rag_pipeline(prompt, indexed_docs, use_rerank=None, chat_history=None):
    """
    Core RAG Orchestration Pipeline entry point.

    Delegates to the LangGraph research graph (src/core/agent) when
    config.AGENTIC_PIPELINE_ENABLED is set; any graph failure falls back to
    the legacy in-file pipeline below, which remains byte-for-byte the
    pre-upgrade behavior.
    """
    if not indexed_docs:
        return {
            "intent": "STANDARD_CHAT",
            "answer": "No documents are currently indexed in your database. Please upload and index a PDF paper first!",
            "image_results": [],
            "search_results": [],
            "is_special_intent": True
        }

    if config.AGENTIC_PIPELINE_ENABLED:
        try:
            from src.core.agent.graph import execute_agentic_pipeline
            result = execute_agentic_pipeline(prompt, indexed_docs, use_rerank, chat_history)
            if result is not None:
                return result
            print("Falling back to legacy pipeline after graph failure.")
        except Exception as e:
            print(f"Falling back to legacy pipeline after graph error: {e}")

    return _execute_legacy_pipeline(prompt, indexed_docs, use_rerank, chat_history)


def _execute_legacy_pipeline(prompt, indexed_docs, use_rerank=None, chat_history=None):
    """
    Pre-upgrade pipeline, retained verbatim as the fallback path.
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
        try:
            registry = get_registry()
            docs_data = []
            for target_doc in targets:
                doc_chunks = [chunk for chunk in registry if chunk["source_file"] == target_doc]
                if doc_chunks:
                    docs_data.append({"name": target_doc, "chunks": doc_chunks})
                    
            if len(docs_data) >= 2:
                answer = generate_comparison(docs_data)
            else:
                answer = f"Error: Could only retrieve chunks for {len(docs_data)} of the requested documents. (Need at least 2)."
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
        # Standard RAG Q&A with Query Expansion & Reciprocal Rank Fusion (RRF)
        RRF_K = 60
        
        # 1. Expand query to generate 5 alternative queries
        expanded_queries = generate_similar_queries(prompt)
        all_queries = [prompt] + [q for q in expanded_queries if q != prompt]
        print(f"Executing RRF Retrieval with expanded queries: {all_queries}")
        
        # 2. Retrieve text chunks for all queries (single batched embedding pass)
        from src.core.embedder import embed_queries
        query_vectors = embed_queries(all_queries)
        query_text_rankings = []
        for q, vec in zip(all_queries, query_vectors):
            q_res = search_store(q, rerank=use_rerank, query_vector=vec)
            if targets:
                q_res = [res for res in q_res if res["chunk"]["source_file"] in targets]
            query_text_rankings.append(q_res)
            
        # Compute RRF scores for text chunks
        text_rrf_scores = {}
        chunk_objects = {}
        for rankings in query_text_rankings:
            for rank_0, res in enumerate(rankings):
                chunk = res["chunk"]
                key = (chunk["source_file"], chunk["page"], chunk["text"])
                rank = rank_0 + 1
                if key not in text_rrf_scores:
                    text_rrf_scores[key] = 0.0
                    chunk_objects[key] = chunk
                text_rrf_scores[key] += 1.0 / (RRF_K + rank)
                
        # Sort text chunks by RRF score descending and take top 5
        sorted_text_keys = sorted(text_rrf_scores.keys(), key=lambda k: text_rrf_scores[k], reverse=True)
        search_results = []
        for key in sorted_text_keys[:5]:
            search_results.append({
                "score": text_rrf_scores[key],
                "chunk": chunk_objects[key]
            })
            
        # 3. Retrieve visual chunks for all queries
        query_image_rankings = []
        for q in all_queries:
            q_res = search_image_store(q)
            if targets:
                q_res = [res for res in q_res if res["chunk"]["source_file"] in targets]
            query_image_rankings.append(q_res)
            
        # Compute RRF scores for visual chunks
        image_rrf_scores = {}
        image_objects = {}
        for query_idx, rankings in enumerate(query_image_rankings):
            for rank_0, res in enumerate(rankings):
                chunk = res["chunk"]
                key = (chunk["source_file"], chunk["page"], chunk["image_path"])
                rank = rank_0 + 1
                if key not in image_rrf_scores:
                    image_rrf_scores[key] = 0.0
                    image_objects[key] = chunk
                    
                # RRF Rank Score contribution
                contrib = 1.0 / (RRF_K + rank)
                
                # Apply a massive boost if this chunk explicitly matched the figure/table reference in the original query
                # (search_image_store marks explicit reference hits with the "explicit" flag)
                if query_idx == 0 and res.get("explicit") is True:
                    contrib += 2.0

                image_rrf_scores[key] += contrib

        # Sort visual chunks by RRF score descending, capped by config
        sorted_img_keys = sorted(image_rrf_scores.keys(), key=lambda k: image_rrf_scores[k], reverse=True)
        image_results = []
        for key in sorted_img_keys[:config.K_IMAGE_RETRIEVAL]:
            image_results.append({
                "score": image_rrf_scores[key],
                "chunk": image_objects[key]
            })
            
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
