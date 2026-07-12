
import os 
import faiss
import json
from sentence_transformers import SentenceTransformer
from google.genai import types
from google import genai

from dotenv import load_dotenv

load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")


embedding_model= SentenceTransformer(
    "BAAI/bge-large-en-v1.5"
)


index= faiss.read_index(
    "../storage/embedding_db.faiss"
)

#print("vectors in faiss:",index.ntotal)


with open("../storage/chunks.json","r") as f:
    all_chunks=json.load(f)

#print("chunks loaded :" ,(len(all_chunks)))




user_query =input("whats the query?:")

query_embedding=embedding_model.encode(
    user_query,
    convert_to_numpy=True,
    normalize_embeddings=True   
                              )

query_embedding=query_embedding.reshape(1,-1)

print(query_embedding.shape)

scores,indices=index.search(query_embedding,k=5)

print(indices)

print()

print(scores)

system_prompt = """ 
You are a research assistant.

Answer questions using ONLY the provided context.

If the answer is not present in the context, say:

'I could not find the answer in the input documents.'

Be concise and factual.
"""

context = ""

for idx in indices[0]:

      context += f"""
=== source Page {all_chunks[idx]["page"]}========= 
{all_chunks[idx]["text"]}
-----------------------------
"""


      for table in all_chunks[idx]["tables"]:
        context += f"""
TABLE CONTENT :
{table["text"]}
----------------------------
TABLE :
{table["html"]}       

"""
      context+= """
------------------------
"""




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


Answer the question using only  retreived context.
If the answer cannot be found in the retrieved context, say:

"I could not find the answer in the input documents."
"""





print(prompt[:2000])


client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents= prompt,
    config=types.GenerateContentConfig(
        temperature=0.5
    )
)

print(response.text)

print()

print("phase successfully completed")

