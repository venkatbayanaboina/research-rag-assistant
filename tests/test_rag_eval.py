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
    def ensure_ollama_running(self):
        import requests
        import subprocess
        import time
        try:
            # Check if already running
            res = requests.get("http://localhost:11434/api/tags", timeout=2)
            if res.status_code == 200:
                return True
        except Exception:
            pass

        print("SYSTEM LOG: Ollama is not running. Attempting to start local Ollama server...")
        try:
            # Start Ollama server in background
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait up to 10 seconds for it to start and bind
            for i in range(10):
                time.sleep(1)
                try:
                    res = requests.get("http://localhost:11434/api/tags", timeout=1)
                    if res.status_code == 200:
                        print("SYSTEM LOG: Local Ollama server started successfully.")
                        return True
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ Failed to start Ollama automatically: {e}")
        return False

    def __init__(self, model_name="llama3"):
        self.model_name = model_name
        self.use_ollama = False
        
        # Ensure Ollama server is active
        self.ensure_ollama_running()
        
        # Check if Ollama local server is active
        import requests
        try:
            res = requests.get("http://localhost:11434/api/tags", timeout=2)
            if res.status_code == 200:
                pulled_models = [m["name"] for m in res.json().get("models", [])]
                
                # Priority preference matching
                preferences = ["qwen3", "qwen2.5", "llama3.3", "llama3.1", "llama3.2", "mistral", "gemma3", "llama3"]
                selected_model = None
                for pref in preferences:
                    for name in pulled_models:
                        if name.split(":")[0] == pref or name == pref:
                            selected_model = name
                            break
                    if selected_model:
                        break
                
                if selected_model:
                    self.use_ollama = True
                    self.model_name = selected_model
                elif pulled_models:
                    self.use_ollama = True
                    self.model_name = pulled_models[0]
                
                print(f"SYSTEM LOG: Local Ollama server detected. Using local model: {self.model_name}")
        except Exception:
            pass

    def load_model(self):
        return self

    def clean_json_verdicts(self, json_str: str) -> str:
        """Self-healing JSON cleaner to normalize typos like 'verdet' or 'verdit' to 'verdict'."""
        import json
        try:
            data = json.loads(json_str)
            if isinstance(data, dict):
                cleaned_data = {}
                # 1. Clean root-level keys
                for k, v in data.items():
                    k_clean = k.lower().strip()
                    if k_clean in ("verdict", "verdicts", "verdet", "verdets", "verdit", "verdits"):
                        cleaned_data["verdicts"] = v
                    else:
                        cleaned_data[k] = v
                
                # 2. Clean items inside the verdicts list
                if "verdicts" in cleaned_data and isinstance(cleaned_data["verdicts"], list):
                    cleaned_verdicts = []
                    for item in cleaned_data["verdicts"]:
                        if isinstance(item, dict):
                            cleaned_item = {}
                            for k, v in item.items():
                                k_clean = k.lower().strip()
                                if k_clean in ("verdict", "verdet", "verdit"):
                                    cleaned_item["verdict"] = v
                                else:
                                    cleaned_item[k] = v
                            cleaned_verdicts.append(cleaned_item)
                        else:
                            cleaned_verdicts.append(item)
                    cleaned_data["verdicts"] = cleaned_verdicts
                
                return json.dumps(cleaned_data)
        except Exception as e:
            print(f"⚠️ JSON self-healing failed to parse: {e}")
        
        # Fallback to regex-like string replacements
        modified = json_str.replace('"verdet":', '"verdict":').replace('"verdit":', '"verdict":')
        modified = modified.replace('"verdets":', '"verdicts":').replace('"verdits":', '"verdicts":')
        return modified

    def generate(self, prompt: str) -> str:
        if self.use_ollama:
            import requests
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_ctx": 16384,
                        "num_predict": 2048
                    }
                }
                
                # Check if prompt requests JSON
                if "json" in prompt.lower() or "schema" in prompt.lower() or "{" in prompt:
                    payload["format"] = "json"
                    
                response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
                if response.status_code == 200:
                    raw_res = response.json()["response"]
                    return self.clean_json_verdicts(raw_res)
            except Exception as e:
                print(f"⚠️ Local Ollama query failed: {e}. Falling back to cloud LLMGate...")
                
        # Fallback to cloud gateways (Gemini/OpenRouter)
        return generate_via_gate(prompt, temperature=0.1)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def generate_with_schema(self, prompt: str, schema) -> str:
        """
        Specialized structured JSON generator for DeepEval.
        Instructs the local Ollama instance to strictly output schema-conforming JSON keys.
        """
        if self.use_ollama:
            import requests
            import json
            try:
                # Compile a prompt instruction mentioning the expected JSON schema structure
                schema_instructions = f"Your output MUST strictly conform to this JSON schema:\n{json.dumps(schema.schema(), indent=2)}\n\nPrompt:\n{prompt}"
                
                payload = {
                    "model": self.model_name,
                    "prompt": schema_instructions,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.1,
                        "num_ctx": 16384,
                        "num_predict": 2048
                    }
                }
                
                response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
                if response.status_code == 200:
                    raw_res = response.json()["response"]
                    return self.clean_json_verdicts(raw_res)
            except Exception as e:
                print(f"⚠️ Local Ollama structured query failed: {e}. Falling back to cloud LLMGate...")
                
        # Fallback to cloud Gatekeeper routing
        return generate_via_gate(prompt, temperature=0.1, response_mime_type="application/json")

    async def a_generate_with_schema(self, prompt: str, schema) -> str:
        return self.generate_with_schema(prompt, schema)

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
