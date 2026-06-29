import os
import sys
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.generator import execute_rag_pipeline
from src.core.generator.gates import generate_via_gate
from src.core.vector_store import get_indexed_documents

GOLDEN_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "golden_dataset.json")

# Dynamic Judge wrapper
class LLMGateJudge(DeepEvalBaseLLM):
    """
    Custom DeepEval Judge Model wrapper.
    Auto-detects if a local Ollama server is active (e.g. running Llama3 on Colab GPU).
    If present, routes evaluations locally for free. Otherwise, falls back to the cloud LLMGate.
    """
    def __init__(self, model_name="llama3"):
        self.model_name = model_name
        self.use_ollama = False
        
        # Check if Ollama local server is active
        import requests
        try:
            res = requests.get("http://localhost:11434", timeout=1)
            if res.status_code == 200:
                self.use_ollama = True
                print("SYSTEM LOG: Local Ollama server detected. Using local Llama3 as the evaluation judge.")
        except Exception:
            pass

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        if self.use_ollama:
            import requests
            try:
                payload = {
                    "model": "llama3",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                }
                response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
                if response.status_code == 200:
                    return response.json()["response"]
            except Exception as e:
                print(f"⚠️ Local Ollama query failed: {e}. Falling back to cloud LLMGate...")
                
        # Fallback to cloud gateways (Gemini/OpenRouter)
        return generate_via_gate(prompt, temperature=0.1)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

def load_golden_cases():
    """Loads generated test cases from JSON, falling back to a default if not found."""
    if os.path.exists(GOLDEN_DATASET_PATH):
        try:
            with open(GOLDEN_DATASET_PATH, "r") as f:
                cases = json.load(f)
                if cases:
                    return cases
        except Exception:
            pass
    return [{
        "query": "what is the difference between trojan paper and the spoofing attack?",
        "ground_truth": None,
        "source_file": None
    }]

@pytest.mark.parametrize("case", load_golden_cases())
def test_rag_faithfulness_and_relevancy(case):
    """
    Automated RAG evaluation test. Asserts that the synthesized answer is
    faithful to the retrieved context and relevant to the user query.
    """
    indexed_docs = get_indexed_documents()
    if not indexed_docs:
        pytest.skip("No documents indexed in FAISS. Skipping evaluation test.")
        
    query = case["query"]
    target_doc = case.get("source_file")
    
    # 1. Execute RAG Pipeline
    result = execute_rag_pipeline(prompt=query, indexed_docs=indexed_docs)
    
    # 2. Extract context chunks text
    retrieval_context = [res["chunk"]["text"] for res in result["search_results"]]
    
    # If it was routed as comparison/special intent, we load raw comparison text as context
    if not retrieval_context and result["is_special_intent"]:
        from src.core.vector_store import get_registry
        registry = get_registry()
        retrieval_context = [
            chunk["text"] for chunk in registry 
            if chunk["source_file"] == target_doc or chunk["source_file"] in indexed_docs
        ]

    # Ensure context is not empty to avoid DeepEval assertion crashes
    if not retrieval_context:
        retrieval_context = ["No document chunks were fetched. Database is empty or filtering blocked matches."]
        
    # 3. Construct LLMTestCase
    test_case = LLMTestCase(
        input=query,
        actual_output=result["answer"],
        retrieval_context=retrieval_context,
        expected_output=case.get("ground_truth")
    )
    
    # 4. Instantiate custom judge and metrics with a 0.6 threshold
    judge = LLMGateJudge()
    faithfulness_metric = FaithfulnessMetric(threshold=0.6, model=judge)
    relevancy_metric = AnswerRelevancyMetric(threshold=0.6, model=judge)
    
    # 5. Assert quality threshold criteria
    assert_test(test_case, [faithfulness_metric, relevancy_metric])
