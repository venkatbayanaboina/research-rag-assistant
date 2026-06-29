import os
import sys
import time
from google import genai
from google.genai import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
import config
from src.core.generator.gates.base import BaseModelGate, GeminiQuotaExhaustedError

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
            except Exception:
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
