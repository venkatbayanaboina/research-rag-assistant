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

    def generate(self, prompt, system_instruction=None, response_mime_type=None, temperature=0.3):
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
