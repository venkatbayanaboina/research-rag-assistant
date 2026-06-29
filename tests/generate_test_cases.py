import os
import sys
import json
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.core.vector_store import get_registry
from src.core.generator.gates import generate_via_gate

GOLDEN_DATASET_PATH = os.path.join(config.BASE_DIR, "tests", "golden_dataset.json")

def generate_dataset(num_questions_per_doc=3):
    """
    Scans the chunks registry, samples text, and calls LLMGate 
    to synthesize a Golden Q&A evaluation dataset.
    """
    registry = get_registry()
    if not registry:
        print("\n❌ No indexed document chunks found. Please ingest a PDF first (e.g. python main.py ingest <file>).")
        return

    # Group chunks by source file
    doc_chunks = {}
    for chunk in registry:
        src = chunk["source_file"]
        if src not in doc_chunks:
            doc_chunks[src] = []
        doc_chunks[src].append(chunk)

    # Auto-detect local Ollama server and pulled model
    use_ollama = False
    ollama_model = "llama3"
    import requests
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=2)
        if res.status_code == 200:
            use_ollama = True
            pulled_models = [m["name"] for m in res.json().get("models", [])]
            # Strip tags for comparison
            short_names = [name.split(":")[0] for name in pulled_models]
            
            if "llama3" in short_names:
                ollama_model = "llama3"
            elif "llama3.2" in short_names:
                ollama_model = "llama3.2"
            elif pulled_models:
                ollama_model = pulled_models[0] # Fallback to first available model
                
            print(f"SYSTEM LOG: Local Ollama server detected. Using local model: {ollama_model}")
    except Exception:
        pass

    golden_cases = []

    print(f"\nScanning registry. Found {len(doc_chunks)} indexed documents...")
    
    for doc_name, chunks in doc_chunks.items():
        print(f"Generating Q&A pairs for: {doc_name}...")
        
        # Filter chunks that have a substantial text length (e.g. > 800 chars)
        good_chunks = [c for c in chunks if len(c["text"].strip()) > 800]
        if not good_chunks:
            good_chunks = chunks
            
        # Sample chunks randomly to generate diverse questions
        sampled_chunks = random.sample(good_chunks, min(num_questions_per_doc, len(good_chunks)))
        
        for idx, chunk in enumerate(sampled_chunks):
            chunk_text = chunk["text"]
            
            system_instruction = (
                "You are an academic test suite generator. Based on the provided document text, "
                "generate exactly 1 realistic user search query that a researcher would ask, "
                "and the corresponding precise, factual Ground Truth answer. "
                "The ground truth must be derived strictly and only from the provided text.\n"
                "Return your decision strictly in JSON format matching this schema:\n"
                "{\n"
                '  "query": "the user search query",\n'
                '  "ground_truth": "precise factual answer"\n'
                "}"
            )
            
            prompt = f"DOCUMENT SECTION:\n{chunk_text}"
            
            try:
                if use_ollama:
                    # Query local Ollama
                    import requests
                    combined_prompt = f"{system_instruction}\n\nDOCUMENT SECTION:\n{chunk_text}"
                    payload = {
                        "model": ollama_model,
                        "prompt": combined_prompt,
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0.3}
                    }
                    response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
                    if response.status_code == 200:
                        response_text = response.json()["response"]
                    else:
                        raise RuntimeError(f"Ollama server returned status {response.status_code}")
                else:
                    # Call HA cloud gatekeeper
                    response_text = generate_via_gate(
                        prompt=prompt,
                        temperature=0.3,
                        system_instruction=system_instruction,
                        response_mime_type="application/json"
                    )
                
                # Clean up potential markdown wrapper from response text
                cleaned_content = response_text.strip()
                if cleaned_content.startswith("```json"):
                    cleaned_content = cleaned_content[7:]
                if cleaned_content.startswith("```"):
                    cleaned_content = cleaned_content[3:]
                if cleaned_content.endswith("```"):
                    cleaned_content = cleaned_content[:-3]
                cleaned_content = cleaned_content.strip()

                qa_pair = json.loads(cleaned_content)
                if "query" in qa_pair and "ground_truth" in qa_pair:
                    golden_cases.append({
                        "query": qa_pair["query"],
                        "ground_truth": qa_pair["ground_truth"],
                        "source_file": doc_name
                    })
                    print(f"  * Generated Case {idx+1}: '{qa_pair['query'][:60]}...'")
            except Exception as e:
                print(f"  ⚠️ Failed to generate Q&A pair: {e}")
                continue

    if golden_cases:
        # Create tests folder if not exists
        os.makedirs(os.path.dirname(GOLDEN_DATASET_PATH), exist_ok=True)
        with open(GOLDEN_DATASET_PATH, "w") as f:
            json.dump(golden_cases, f, indent=2)
        print(f"\n✅ Successfully generated {len(golden_cases)} test cases.")
        print(f"💾 Saved Golden Dataset to: {GOLDEN_DATASET_PATH}\n")
    else:
        print("\n❌ Failed to generate any Q&A cases.")

if __name__ == "__main__":
    import sys
    num_questions = 3
    if len(sys.argv) > 1:
        try:
            num_questions = int(sys.argv[1])
        except ValueError:
            print(f"⚠️ Invalid argument '{sys.argv[1]}'. Defaulting to 3 questions per document.")
            
    generate_dataset(num_questions_per_doc=num_questions)
