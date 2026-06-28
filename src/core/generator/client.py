import os
import sys
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Include root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import config

# Load env file
load_dotenv(os.path.join(config.BASE_DIR, ".env"))

_client = None

def get_gemini_client():
    """Initializes and caches the Gemini API client."""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # Check for colab secret fallback
            try:
                from google.colab import userdata
                api_key = userdata.get("GEMINI_API_KEY")
            except ImportError:
                pass
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found. Please check your .env file or Colab Secrets.")
        
        print("Initializing Gemini API Client...")
        
        # Monkey-patch sys.modules to bypass google-genai's buggy environment checks in Streamlit/Colab
        import sys
        ipython_module = sys.modules.get("IPython")
        colab_module = sys.modules.get("google.colab")
        
        if "IPython" in sys.modules:
            del sys.modules["IPython"]
        if "google.colab" in sys.modules:
            del sys.modules["google.colab"]
            
        try:
            _client = genai.Client(api_key=api_key)
        finally:
            # Restore modules to avoid disrupting interactive features
            if ipython_module:
                sys.modules["IPython"] = ipython_module
            if colab_module:
                sys.modules["google.colab"] = colab_module
                
    return _client

def generate_content_with_retry(prompt, is_image_list=False, temperature=0.3):
    """
    Executes a generate_content call to Gemini with an automatic 503 retry loop.
    Supports either a pure text prompt or a list (containing text and PIL Images).
    """
    client = get_gemini_client()
    max_retries = 3
    
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature
                )
            )
            return response.text
        except Exception as e:
            err_msg = str(e).upper()
            is_503 = "503" in err_msg or "UNAVAILABLE" in err_msg
            is_429 = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "RATE" in err_msg
            
            if (is_503 or is_429) and attempt < max_retries:
                sleep_time = 5 if is_429 else 2
                reason = "Rate limit (429)" if is_429 else "Server busy (503)"
                print(f"Gemini API: {reason}. Retrying in {sleep_time} seconds (Attempt {attempt}/{max_retries})...")
                time.sleep(sleep_time)
                continue
                
            # If all Gemini retries fail, check for OpenRouter fallback keys
            or_key = os.getenv("OPENROUTER_API_KEY")
            if not or_key:
                try:
                    from google.colab import userdata
                    or_key = userdata.get("OPENROUTER_API_KEY")
                except ImportError:
                    pass
                    
            if or_key:
                print("Gemini API failed or exhausted quota. Triggering OpenRouter fallback...")
                try:
                    from src.core.generator.openrouter import generate_content_via_openrouter
                    # Map config.GEMINI_MODEL_NAME to OpenRouter model equivalents
                    or_model = "google/gemini-2.5-flash"
                    if "pro" in config.GEMINI_MODEL_NAME.lower():
                        or_model = "google/gemini-2.5-pro"
                        
                    return generate_content_via_openrouter(prompt, model_name=or_model, temperature=temperature)
                except Exception as or_err:
                    print(f"OpenRouter fallback also failed: {or_err}")
                    
            raise e
