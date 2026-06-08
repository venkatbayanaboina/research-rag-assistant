import os

from google import genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

api_key =os.getenv("GEMINI_API_KEY")

client=genai.Client(api_key=api_key)

text ="Gradient descent is an optimisation algorithm"

model=SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embedding=model.encode(text)
print("Embedding dimensions :", len(embedding))
print()
print("First 10 valuse")
print(embedding[:10])
