import os

from google import genai
from dotenv import load_dotenv


load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")

client =genai.Client(api_key=api_key)

chat= client.chats.create(
    model="gemini-3.5-flash"
)
#history =[]

while True:
    user_input=input("you: ")
    if user_input.lower() in ["quit","exit"]:
        print ("chat ended.")
        break
    #history.append({"role":"user","parts" :[user_input]} ,)
    response= chat.send_message(user_input)

  
    print(response.text)
    #history.append(
     #   {"role":"model", "parts":[response.text]}
    

