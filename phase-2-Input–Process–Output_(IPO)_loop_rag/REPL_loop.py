import os

from google import genai
from dotenv import load_dotenv


load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")

client =genai.Client(api_key=api_key)

while True:
    user_input=input("you: ")
    if user_input.lower() in ["quit","exit"]:
        print ("chat ended.")
        break
    response= client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_input
    )    

    print(response.text)

