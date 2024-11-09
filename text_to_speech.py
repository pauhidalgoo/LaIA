import requests
from dotenv import load_dotenv
import os
load_dotenv(".env")

API_URL = os.environ["API_URL"]
HF_TOKEN = os.environ["HF_TOKEN"]

print(API_URL)
headers = {
   "Authorization": f"Bearer {HF_TOKEN}",
}

def query(text):
   data = {"text": text, "voice": 20}
   return requests.post(API_URL, headers=headers, json=data)

response = query("Hola bon dia soc un gat de color verd")

print(response)

with open("output.wav", "wb") as f:
   f.write(response.content)