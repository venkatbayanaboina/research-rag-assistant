import os
import sys
import json
import time
import base64
from io import BytesIO
import requests
from PIL import Image
from google import genai
from google.genai import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import config

class GeminiQuotaExhaustedError(Exception):
    """Raised when the primary Gemini API has completely exhausted its daily request quota."""
    pass

class BaseModelGate:
    def generate(self, prompt, system_instruction=None, response_mime_type=None, temperature=0.3):
        raise NotImplementedError()

class GeminiGate(BaseModelGate):
    _client = None
    _last_api_key = None

    def _get_client(self):
        # Dynamic reload of .env to catch key updates
        from dotenv import load_dotenv
        load_dotenv(os.path.join(config.BASE_DIR, ".env"), override=True)
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            try:
                from google.colab import userdata
                api_key = userdata.get("GEMINI_API_KEY")
            except ImportError:
                pass
                
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found.")
            
        if GeminiGate._client is None or api_key != GeminiGate._last_api_key:
            print("Initializing Gemini API Client...")
            
            # Monkey-patch sys.modules to bypass google-genai's environment checks
            import sys
            ipython_module = sys.modules.get("IPython")
            colab_module = sys.modules.get("google.colab")
            
            if "IPython" in sys.modules:
                del sys.modules["IPython"]
            if "google.colab" in sys.modules:
                del sys.modules["google.colab"]
                
            try:
                GeminiGate._client = genai.Client(api_key=api_key)
                GeminiGate._last_api_key = api_key
            finally:
                if ipython_module:
                    sys.modules["IPython"] = ipython_module
                if colab_module:
                    sys.modules["google.colab"] = colab_module
                    
        return GeminiGate._client

    def generate(self, prompt, system_instruction=None, response_mime_type=None, temperature=0.3):
        client = self._get_client()
        max_retries = 3
        
        gen_config = types.GenerateContentConfig(temperature=temperature)
        if system_instruction:
            gen_config.system_instruction = system_instruction
        if response_mime_type:
            gen_config.response_mime_type = response_mime_type
            
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=config.GEMINI_MODEL_NAME,
                    contents=prompt,
                    config=gen_config
                )
                return response.text
            except Exception as e:
                err_msg = str(e).upper()
                is_503 = "503" in err_msg or "UNAVAILABLE" in err_msg
                is_429 = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "RATE" in err_msg
                
                # Check if it is a daily quota exhaustion (can't be fixed by waiting)
                is_daily_quota = "DAILY" in err_msg or "QUOTA" in err_msg
                
                if is_daily_quota:
                    raise GeminiQuotaExhaustedError(e)
                    
                if (is_503 or is_429) and attempt < max_retries:
                    sleep_time = 10 if is_429 else 2
                    reason = "Rate limit (429)" if is_429 else "Server busy (503)"
                    print(f"Gemini API: {reason}. Retrying in {sleep_time} seconds (Attempt {attempt}/{max_retries})...")
                    time.sleep(sleep_time)
                    continue
                raise e

class OpenRouterGate(BaseModelGate):
    def _encode_pil_image(self, pil_image):
        buffered = BytesIO()
        if pil_image.mode in ("RGBA", "P"):
            pil_image = pil_image.convert("RGB")
        pil_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def generate(self, prompt, system_instruction=None, response_mime_type=None, temperature=0.3):
        # Dynamic reload of .env
        from dotenv import load_dotenv
        load_dotenv(os.path.join(config.BASE_DIR, ".env"), override=True)
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            try:
                from google.colab import userdata
                api_key = userdata.get("OPENROUTER_API_KEY")
            except ImportError:
                pass
                
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set. Cannot execute OpenRouter fallback.")
            
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/venkatbayanaboina/research-rag-assistant",
            "X-Title": "Multi-PDF RAG Assistant"
        }
        
        # Parse prompt contents
        text_prompt = ""
        images = []
        
        if isinstance(prompt, list):
            for item in prompt:
                if isinstance(item, str):
                    text_prompt += item + "\n"
                elif isinstance(item, Image.Image):
                    images.append(item)
        else:
            text_prompt = str(prompt)
            
        messages = []
        if system_instruction:
            messages.append({
                "role": "system",
                "content": system_instruction
            })
            
        user_content = []
        if text_prompt.strip():
            user_content.append({
                "type": "text",
                "text": text_prompt
            })
            
        for img in images:
            b64_img = self._encode_pil_image(img)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_img}"
                }
            })
            
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        payload = {
            "model": "openrouter/free",
            "messages": messages,
            "temperature": temperature
        }
        
        if response_mime_type == "application/json":
            payload["response_format"] = {"type": "json_object"}
            
        print("Sending request to OpenRouter (Model: openrouter/free)...")
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            res_data = response.json()
            try:
                return res_data["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                raise ValueError(f"Unexpected response structure from OpenRouter: {res_data}")
        else:
            raise RuntimeError(f"OpenRouter API returned error status {response.status_code}: {response.text}")

# Global controller state for dynamic gate routing
_active_gate_instance = None

def get_llm_gate():
    """Returns the current active LLM Gate instance."""
    global _active_gate_instance
    if _active_gate_instance is None:
        _active_gate_instance = GeminiGate()
    return _active_gate_instance

def switch_active_gate_to_openrouter():
    """Switches the global LLM Gate permanently to OpenRouter for this runtime session."""
    global _active_gate_instance
    if not isinstance(_active_gate_instance, OpenRouterGate):
        print("SYSTEM LOG: Switching active LLM gate to OpenRouter Gate.")
        _active_gate_instance = OpenRouterGate()

def generate_via_gate(prompt, is_image_list=False, temperature=0.3, system_instruction=None, response_mime_type=None):
    """
    Unified gatekeeper function that coordinates query execution across active gates.
    If the Gemini gate fails or reports quota exhaustion, it permanently switches to the OpenRouter gate.
    """
    gate = get_llm_gate()
    
    try:
        return gate.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            response_mime_type=response_mime_type,
            temperature=temperature
        )
    except (GeminiQuotaExhaustedError, Exception) as e:
        # If the gate is already OpenRouterGate, don't try to switch/retry
        if isinstance(gate, OpenRouterGate):
            raise e
            
        # Switch permanently to OpenRouter for subsequent requests
        print(f"Gemini Gate failed with exception: {e}. Switching active gate to OpenRouter...")
        switch_active_gate_to_openrouter()
        
        # Retry the failed request immediately using OpenRouterGate
        new_gate = get_llm_gate()
        try:
            return new_gate.generate(
                prompt=prompt,
                system_instruction=system_instruction,
                response_mime_type=response_mime_type,
                temperature=temperature
            )
        except Exception as or_err:
            raise RuntimeError(
                f"Gemini Gate failed ({e}). OpenRouter Gate also failed: {or_err}"
            ) from e
