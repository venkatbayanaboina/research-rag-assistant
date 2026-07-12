import os
import sys
import base64
import requests
from io import BytesIO
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
import config
from src.core.generator.gates.base import BaseModelGate

class OpenRouterGate(BaseModelGate):
    def _encode_pil_image(self, pil_image):
        buffered = BytesIO()
        if pil_image.mode in ("RGBA", "P"):
            pil_image = pil_image.convert("RGB")
        pil_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def generate(self, prompt, system_instruction=None, response_mime_type=None, temperature=0.0):
        import time
        print("Pausing 4.5 seconds to respect OpenRouter rate limits...")
        time.sleep(4.5)
        
        # Dynamic reload of .env
        from dotenv import load_dotenv
        load_dotenv(os.path.join(config.BASE_DIR, ".env"), override=True)
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            try:
                from google.colab import userdata
                api_key = userdata.get("OPENROUTER_API_KEY")
            except Exception:
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
            "messages": messages,
            "temperature": temperature
        }
        
        if response_mime_type == "application/json":
            payload["response_format"] = {"type": "json_object"}
            
        # Rotate through highly capable, permanently free models on OpenRouter
        models = [
            "meta-llama/llama-3.2-3b-instruct:free",
            "openrouter/free"
        ]
        
        last_error = None
        for model in models:
            payload["model"] = model
            print(f"Sending request to OpenRouter (Model: {model})...")
            
            try:
                response = requests.post(url, headers=headers, json=payload)
                if response.status_code == 200:
                    res_data = response.json()
                    content = res_data["choices"][0]["message"]["content"].strip()
                    
                    # Clean markdown code blocks if the model wrapped JSON
                    cleaned_content = content
                    if cleaned_content.startswith("```json"):
                        cleaned_content = cleaned_content[7:]
                    if cleaned_content.startswith("```"):
                        cleaned_content = cleaned_content[3:]
                    if cleaned_content.endswith("```"):
                        cleaned_content = cleaned_content[:-3]
                    cleaned_content = cleaned_content.strip()
                    
                    # Guardrail: Check if a safety classification model (e.g. Llama-Guard) intercepted the request
                    if "user safety: safe" in cleaned_content.lower():
                        print(f"⚠️ OpenRouter Model {model} request was safety-filtered (returned '{content}'). Rotating model...")
                        continue
                        
                    # Guardrail: If JSON requested, verify that the returned string is valid JSON
                    if response_mime_type == "application/json":
                        try:
                            import json
                            json.loads(cleaned_content)
                        except json.JSONDecodeError:
                            print(f"⚠️ OpenRouter Model {model} failed to return valid JSON. Rotating model...")
                            continue
                            
                    return content
                else:
                    print(f"⚠️ OpenRouter Model {model} failed with status {response.status_code}: {response.text}")
                    last_error = f"API Status {response.status_code}: {response.text}"
            except Exception as e:
                print(f"⚠️ OpenRouter Model {model} raised exception: {e}")
                last_error = e
                continue
                
        raise RuntimeError(f"All OpenRouter candidate models failed. Last error: {last_error}")
