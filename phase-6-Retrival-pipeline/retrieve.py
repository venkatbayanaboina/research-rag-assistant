import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Resolve paths relative to this script's directory for robust execution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(SCRIPT_DIR, "../storage/embedding_db.faiss")
CHUNKS_PATH = os.path.join(SCRIPT_DIR, "../storage/chunks.json")

# Initialize embedding model
embedding_model = SentenceTransformer(
    "BAAI/bge-large-en-v1.5"
)

# Load index and chunks
index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "r") as f:
    all_chunks = json.load(f)

# Get query from user
user_query = input("whats the query?: ")

# Prepend the query instruction prefix required by BGE Large models for asymmetric search
instruction = "Represent this sentence for searching relevant passages: "
query_embedding = embedding_model.encode(
    instruction + user_query,
    convert_to_numpy=True,
    normalize_embeddings=True   
)

query_embedding = query_embedding.reshape(1, -1)

# Retrieve top 5 closest chunks
scores, indices = index.search(query_embedding, k=5)

print("\nRetrieved Indices:", indices[0])
print("Retrieval Scores:", scores[0])

system_prompt = """ 
You are a research assistant.

Answer questions using ONLY the provided context.

If the answer is not present in the context, say:
'I could not find the answer in the input documents.'

Be concise and factual.
"""

context = ""
for idx in indices[0]:
    context += f"\n=== source Page {all_chunks[idx]['page']}=========\n"
    context += all_chunks[idx]["text"]
    context += "\n-----------------------------\n"
    
    # Safely handle tables/images if present
    for table in all_chunks[idx].get("tables", []):
        context += f"\nTABLE CONTENT:\n{table.get('text', '')}\n"
        context += f"TABLE HTML:\n{table.get('html', '')}\n----------------------------\n"

prompt = f"""
{system_prompt}

=============================
  QUESTION
=============================
{user_query}

=============================  
  RETRIEVED CONTEXT 
=============================
{context}

=============================
  TASK 
=============================     
Answer the question using only the retrieved context.
If the answer cannot be found in the retrieved context, say:
"I could not find the answer in the input documents."
"""

print("\n--- Sending Prompt to Gemini ---")
client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        temperature=0.5
    )
)

print("\n============================= GEMINI ANSWER =============================")
print(response.text)
print("=========================================================================\n")
print("phase successfully completed")
