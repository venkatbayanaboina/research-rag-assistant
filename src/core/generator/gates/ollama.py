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
    def __init__(self):
        self.model_name = "llama3"
        self.url = "http://localhost:11434/api/generate"
        
        # Check active pulled model names
        try:
            res = requests.get("http://localhost:11434/api/tags", timeout=2)
            if res.status_code == 200:
                pulled_models = [m["name"] for m in res.json().get("models", [])]
                short_names = [name.split(":")[0] for name in pulled_models]
                
                if "llama3" in short_names:
                    self.model_name = "llama3"
                elif "llama3.2" in short_names:
                    self.model_name = "llama3.2"
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
            "options": {"temperature": temperature}
        }
        
        if response_mime_type == "application/json":
            payload["format"] = "json"

        print(f"Sending request to local Ollama (Model: {self.model_name})...")
        
        response = requests.post(self.url, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            raise RuntimeError(f"Ollama server returned status code {response.status_code}: {response.text}")
