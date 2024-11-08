import requests
from dotenv import load_dotenv
import os
load_dotenv(".env")

API_URL = os.environ["API_URL"]
HF_TOKEN = os.environ["HF_TOKEN"]


headers = {
   "Authorization": f"Bearer {HF_TOKEN}",
}

def query(text):
   data = {"text": text, "voice": 20}
   return requests.post(API_URL, headers=headers, json=data)

response = query("cai gei cai gei cai gei")

print(response)

with open("output.wav", "wb") as f:
   f.write(response.content)