import os
import sys
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.core.generator.client import generate_content_with_retry

def generate_answer(query, search_results, image_results=None, chat_history=None):
    """
    Synthesizes a response using Gemini based on retrieved context chunks,
    optional retrieved visual diagrams, and chat history.
    """
    # 1. Construct context text
    context = ""
    if search_results:
        for idx, res in enumerate(search_results):
            chunk = res["chunk"]
            context += f"\n\n[Source {idx+1} - {chunk['source_file']} (Page {chunk['page']})]:\n{chunk['text']}"
            
    # 2. Format chat history
    history_str = ""
    if chat_history:
        history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
        
    # 3. Compile prompt
    prompt = f"""You are an advanced scientific research assistant. Answer the user's question by grounding your response in the retrieved context documents and diagram images.

=============================
  CONVERSATION HISTORY
=============================
{history_str}

=============================
  RETRIEVED CONTEXT
=============================
{context}

=============================
  USER QUESTION
=============================
{query}

=============================
  GENERATION RULES
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

    print("Synthesizing answer with Gemini...")
    return generate_content_with_retry(contents, temperature=0.3)
