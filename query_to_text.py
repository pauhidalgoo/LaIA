import requests
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(".env")
WHISPER_URL = os.environ["WHISPER_URL"]
HF_TOKEN = os.environ["HF_TOKEN"]

headers = {
   "Accept": "application/json",
   "Authorization": f"Bearer {HF_TOKEN}",
   "Content-Type": "audio/wav",
}


def query(filename):
   with open(filename, "rb") as f:
       data = f.read()
   response = requests.post(WHISPER_URL, headers=headers, data=data)
   return response.json()

output = query("output.wav")
print(output)