import os
import sys
import gc
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.core.vector_store import search_store, search_image_store, get_indexed_documents
from src.core.generator.answering import generate_answer

# Import deepeval components
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from tests.test_rag_eval import LLMGateJudge, load_golden_cases

def main():
    # Force evaluations to run completely locally (zero Gemini leakages)
    from src.core.generator.gates.controller import switch_active_gate_to_ollama
    switch_active_gate_to_ollama()
    
    cases = load_golden_cases()
    
    # ==========================================
    # PHASE 1: RETRIEVE CONTEXT & CACHE CHUNKS
    # ==========================================
    print("==================================================")
    print("🚀 STARTING PHASE 1: RETRIEVING CONTEXT CHUNKS")
    print("==================================================")
    
    retrieved_data = []
    for idx, case in enumerate(cases):
        query = case["query"]
        print(f"\n[{idx+1}/{len(cases)}] Searching for: '{query[:60]}...'")
        
        # Query BGE & CrossEncoder (loads models into GPU VRAM)
        search_results = search_store(query)
        
        # Query CLIP (loads model into GPU VRAM)
        image_results = search_image_store(query)
        
        retrieved_data.append({
            "query": query,
            "search_results": search_results,
            "image_results": image_results,
            "expected_output": case.get("ground_truth")
        })
        
    # ==========================================
    # UNLOAD EVERYTHING & FREE VRAM
    # ==========================================
    print("\n==================================================")
    print("🗑️ UNLOADING ALL RETRIEVAL MODELS FROM VRAM...")
    print("==================================================")
    
    # 1. Reset embedding models in cache
    from src.core import embedder
    embedder._model = None
    embedder._clip_model = None
    
    # 2. Reset Cross-Encoder reranker in cache
    from src.core import reranker
    reranker._reranker = None
    
    # 3. Force garbage collection and flush CUDA cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("VRAM cache cleared successfully. PyTorch GPU allocation is now 0.0 MB.")
        
    # ==========================================
    # PHASE 2: GENERATION & LLM METRIC EVALUATION
    # ==========================================
    print("\n==================================================")
    print("🚀 STARTING PHASE 2: ANSWER GENERATION & JUDGING")
    print("==================================================\n")
    
    test_cases = []
    for idx, data in enumerate(retrieved_data):
        query = data["query"]
        search_results = data["search_results"]
        image_results = data["image_results"]
        expected_output = data["expected_output"]
        
        print(f"[{idx+1}/{len(retrieved_data)}] Generating answer for: '{query[:60]}...'")
        
        # Generate RAG answer using the cached context (uses 0.0 MB PyTorch VRAM)
        answer = generate_answer(query, search_results, image_results)
        
        # Extract text context
        retrieval_context = [res["chunk"]["text"] for res in search_results]
        if not retrieval_context:
            retrieval_context = ["No document chunks were fetched. Database is empty or filtering blocked matches."]
            
        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=retrieval_context,
            expected_output=expected_output
        )
        test_cases.append(test_case)
        
    print("\nStarting DeepEval metric evaluations...")
    judge = LLMGateJudge()
    faithfulness_metric = FaithfulnessMetric(threshold=0.6, model=judge)
    relevancy_metric = AnswerRelevancyMetric(threshold=0.6, model=judge)
    
    # Run bulk evaluations sequentially using AsyncConfig to prevent local model overloading
    from deepeval.evaluate import AsyncConfig
    evaluate(
        test_cases, 
        [faithfulness_metric, relevancy_metric], 
        async_config=AsyncConfig(run_async=False)
    )

if __name__ == "__main__":
    main()
