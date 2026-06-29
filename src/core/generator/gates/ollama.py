import os
import requests
import base64
from io import BytesIO
from PIL import Image

from src.core.generator.gates.base import BaseModelGate
import config

class OllamaGate(BaseModelGate):
    """
    Local Ollama Model Gate. 
    Allows RAG queries to run completely locally if cloud APIs are rate-limited.
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

    def __init__(self):
        self.model_name = "llama3"
        self.url = "http://localhost:11434/api/generate"
        
        # Ensure Ollama server is active
        self.ensure_ollama_running()
        
        # Check active pulled model names
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
                    self.model_name = selected_model
                elif pulled_models:
                    self.model_name = pulled_models[0]
        except Exception:
            pass

    def generate(self, prompt, system_instruction=None, response_mime_type=None, temperature=0.3):
        # Format the combined prompt since Ollama standard endpoint expects a unified string
        combined_prompt = ""
        if system_instruction:
            combined_prompt += f"{system_instruction}\n\n"
            
        if isinstance(prompt, list):
            for item in prompt:
                if isinstance(item, str):
                    combined_prompt += item + "\n"
        else:
            combined_prompt += str(prompt)

        payload = {
            "model": self.model_name,
            "prompt": combined_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": 16384,
                "num_predict": 2048
            }
        }
        
        if response_mime_type == "application/json":
            payload["format"] = "json"

        print(f"Sending request to local Ollama (Model: {self.model_name})...")
        
        response = requests.post(self.url, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            raise RuntimeError(f"Ollama server returned status code {response.status_code}: {response.text}")
