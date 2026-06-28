import os
import sys
import json
import base64
from io import BytesIO
import requests
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

def encode_pil_image_to_base64(pil_image):
    """Converts a PIL Image object to a base64 encoded JPEG string."""
    buffered = BytesIO()
    # Convert RGBA to RGB if needed to save as JPEG
    if pil_image.mode in ("RGBA", "P"):
        pil_image = pil_image.convert("RGB")
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_content_via_openrouter(contents, model_name="google/gemini-2.5-flash", temperature=0.3, system_instruction=None, json_mode=False):
    """
    Sends a request to the OpenRouter chat completions endpoint.
    Supports text prompts, system instructions, JSON mode, and PIL Image elements.
    """
    # Dynamic reload to pick up key updates without restarting the server
    from dotenv import load_dotenv
    import config
    load_dotenv(os.path.join(config.BASE_DIR, ".env"), override=True)
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        # Check colab fallback
        try:
            from google.colab import userdata
            api_key = userdata.get("OPENROUTER_API_KEY")
        except ImportError:
            pass
            
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set. Cannot run OpenRouter fallback.")
        
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/venkatbayanaboina/research-rag-assistant",
        "X-Title": "Multi-PDF RAG Assistant"
    }
    
    # Check if contents is a single string or a list of mixed content
    text_prompt = ""
    images = []
    
    if isinstance(contents, list):
        for item in contents:
            if isinstance(item, str):
                text_prompt += item + "\n"
            elif isinstance(item, Image.Image):
                images.append(item)
    else:
        text_prompt = str(contents)
        
    # Construct messages list
    messages = []
    
    # 1. Add system instruction if provided
    if system_instruction:
        messages.append({
            "role": "system",
            "content": system_instruction
        })
        
    # 2. Add user contents
    message_content = []
    if text_prompt.strip():
        message_content.append({
            "type": "text",
            "text": text_prompt
        })
    for img in images:
        base64_str = encode_pil_image_to_base64(img)
        message_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_str}"
            }
        })
        
    messages.append({
        "role": "user",
        "content": message_content
    })
        
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature
    }
    
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    
    print(f"Sending request to OpenRouter (Model: {model_name})...")
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        res_data = response.json()
        try:
            return res_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise ValueError(f"Unexpected response structure from OpenRouter: {res_data}")
    else:
        raise RuntimeError(f"OpenRouter API returned error status {response.status_code}: {response.text}")
