import os
import sys
from google import genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image

# Include root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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
        _client = genai.Client(api_key=api_key)
    return _client

def generate_answer(query, search_results, image_results=None, chat_history=None):
    """
    Synthesizes a response using Gemini based on retrieved context chunks,
    optional retrieved visual diagrams, and chat history.
    """
    client = get_gemini_client()
    
    # 1. Construct context text
    context = ""
    for idx, result in enumerate(search_results):
        chunk = result["chunk"]
        score = result["score"]
        context += f"\n=== Source {idx+1} [File: {chunk['source_file']} | Page: {chunk['page']} | Score: {score:.4f}] ===\n"
        context += chunk["text"]
        context += "\n-----------------------------\n"
        
        # Add tables if present
        for table in chunk.get("tables", []):
            context += f"\n[Table Content]\n{table.get('text', '')}\n"
            context += f"[Table HTML]\n{table.get('html', '')}\n----------------------------\n"

    # 2. Format chat history
    history_str = ""
    if chat_history:
        history_str = "=== CONVERSATION HISTORY ===\n"
        for msg in chat_history:
            history_str += f"{msg['role'].upper()}: {msg['content']}\n"
        history_str += "============================\n"

    # 3. Build Prompt
    system_prompt = """You are a helpful research assistant. 
Answer questions using ONLY the provided retrieved text context and any attached diagrams/images.
If the answer cannot be found in the provided context, say:
'I could not find the answer in the input documents.'
Be concise and factual. Do not hallucinate or use external knowledge."""

    prompt = f"""
{system_prompt}

{history_str}

=============================
  QUESTION
=============================
{query}

=============================  
  RETRIEVED CONTEXT 
=============================
{context}

=============================
  TASK 
=============================     
Answer the question using the retrieved text context and any attached images/diagrams.
If the answer cannot be found in the retrieved context, say:
"I could not find the answer in the input documents."
"""

    # 4. Construct content payload list (Text Prompt + PIL Images)
    contents = [prompt]
    if image_results:
        print(f"Attaching {len(image_results)} visual diagrams/tables to Gemini payload...")
        for res in image_results:
            img_chunk = res["chunk"]
            path = img_chunk["image_path"]
            if os.path.exists(path):
                try:
                    contents.append(Image.open(path))
                except Exception as e:
                    print(f"Failed to open image file at {path}: {e}")

    import time
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            print("Synthesizing answer with Gemini...")
            response = client.models.generate_content(
                model=config.GEMINI_MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.3
                )
            )
            return response.text
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e).upper():
                if attempt < max_retries:
                    print(f"Gemini API is busy (503). Retrying in 2 seconds (Attempt {attempt}/{max_retries})...")
                    time.sleep(2)
                    continue
            raise e

def generate_summary(chunks):
    """
    Generates a structured, comprehensive summary of a document
    using the text extracted from its chunks.
    """
    client = get_gemini_client()
    
    # Reconstruct document text
    full_text = "\n\n".join([chunk["text"] for chunk in chunks])
    filename = chunks[0]["source_file"] if chunks else "Document"
    
    prompt = f"""
You are a senior scientific research analyst.
Analyze the following document text extracted from the PDF '{filename}' and write a structured, comprehensive summary.

Your summary must include:
1. **Core Overview**: A brief high-level description of the paper/document's purpose.
2. **Key Findings / Contributions**: Bullet points of the main achievements, models introduced, or results.
3. **Methodology**: Explanation of the methods, architectures, or procedures described.
4. **Conclusion**: Main takeaways or future directions.

Be highly factual, precise, and professional.

=============================
  DOCUMENT TEXT
=============================
{full_text}
"""

    import time
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Generating summary for {filename} with Gemini...")
            response = client.models.generate_content(
                model=config.GEMINI_MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2
                )
            )
            return response.text
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e).upper():
                if attempt < max_retries:
                    print(f"Gemini API is busy (503). Retrying in 2 seconds (Attempt {attempt}/{max_retries})...")
                    time.sleep(2)
                    continue
            raise e
