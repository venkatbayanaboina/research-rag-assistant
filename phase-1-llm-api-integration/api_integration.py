import os
from dotenv import load_dotenv 
from google import genai  # Using the official Google library!

print("1. Program started")

load_dotenv()
print("2. Loaded dotenv")

print("3. Getting API key")
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("API key not found! Make sure GEMINI_API_KEY is in your .env file.")

# Configure the native Google client
client = genai.Client(api_key=api_key)
print("4. Client configured for Google AI Studio")

print("5. Prompt given to model")

# Make the API call using the native syntax
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Explain Retrieval-Augmented Generation simply.'
)

print("\n===__Gemini Response__===")
print(response.text)