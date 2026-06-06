import os

from google import genai
from dotenv import load_dotenv

load_dotenv()


api_key=os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="give me 5 important questions on operation systems for gate exam"
)
print("___gemini response__")
print(response.text)

