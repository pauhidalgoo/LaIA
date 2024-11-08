#pip install openai
from dotenv import load_dotenv
import os
from openai import OpenAI
load_dotenv(".env")

HF_TOKEN = os.environ["HF_TOKEN"]
BASE_URL = os.environ["BASE_URL"]


#pip install openai
client = OpenAI(
       base_url=BASE_URL + "/v1/",
       api_key=HF_TOKEN
   )
messages = [{ "role": "system", "content": "Et dius Laia i ets una assistent guai i útil"}]
messages.append( {"role":"user", "content": "Hola bon dia com et dius?"})
stream = False
chat_completion = client.chat.completions.create(
   model="tgi",
   messages=messages,
   stream=stream,
   max_tokens=1000,
   # temperature=0.1,
   # top_p=0.95,
   # frequency_penalty=0.2,
)
text = ""
if stream:
 for message in chat_completion:
   text += message.choices[0].delta.content
   print(message.choices[0].delta.content, end="")
 print(text)
else:
 text = chat_completion.choices[0].message.content
 print(text)