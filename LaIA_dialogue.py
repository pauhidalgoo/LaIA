#pip install openai
from dotenv import load_dotenv
import os
from openai import OpenAI


class LaIA_dialogue:
      def __init__(self, text='',messages=[], prompt=""):
         assert text, "Text is required"
         load_dotenv(".env")
         self.messages = messages
         self.HF_TOKEN = os.environ["HF_TOKEN"]
         self.BASE_URL = os.environ["BASE_URL"]
         self._client()

         self.text = text
         if prompt:
            self.prompt = prompt
         else:
            self.prompt = """Genera un diàleg en català entre el Cai i la LaIA sobre un tema mencionat més tard. El diàleg ha de ser breu, informatiu i amb un to amable, seguint aquestes instruccions:

Rol de Cai: Comença amb una pregunta general sobre el tema i, si és necessari, fa una o dues intervencions molt breus després. El seu paper és reactiu, i no fa preguntes detallades. En acabar, el Cai pot agrair la informació amb una frase breu.

Rol de LaIA: Respon de manera completa i anticipa possibles dubtes del Cai. Inclou informació rellevant (funcionament, requisits, terminis, documents necessaris, explicacions, etc.) en una sola resposta extensa, de manera que el Cai no necessiti fer més preguntes. La resposta ha de ser clara i fàcil de seguir.

To i estructura: La LaIA ha de sonar propera, informativa i accessible, com si parlés en una conversa natural. Les respostes han de ser coherents i detallades, cobrint tots els aspectes importants per evitar preguntes addicionals.

Finalització: La conversa ha de ser curta. La LaIA acaba amb una frase de comiat amable, assegurant que el Cai té tota la informació necessària.

El format ha de ser el següent:

Cai: Pregunta
LaIA: Resposta
Cai: Intervenció breu
LaIA: Resposta

Ha de ser una interacció curta amb un objectiui de crear una conversa breu on el Cai només fa una pregunta inicial general i, com a màxim, una o dues intervencions breus de seguiment. La LaIA cobreix tota la informació rellevant en la seva resposta inicial, fent que el diàleg sembli complet i informatiu sense necessitat de més preguntes.
"""

      def _client(self):
         """
         Create an OpenAI client
         """
         self.client = OpenAI(
               base_url=self.BASE_URL + "/v1/",
               api_key=self.HF_TOKEN
             )
         
      def create_dialogue(self):
         """
         Function to create a dialogue given a text and a prompt.
         
         Returns:
         str: dialogue
         """
         self.messages.append({"role":"user", "content": f'{self.prompt}'})
         self.messages.append( {"role":"user", "content": """A continuació tens la informació per tal de poder crear el diàleg: """})
         self.messages.append( {"role":"user", "content": f'{self.text}'})
         stream = False
         chat_completion = self.client.chat.completions.create(
            model="tgi",
            messages=self.messages,
            stream=stream,
            max_tokens=1000,
            temperature=0.1,
            top_p=0.95,
            frequency_penalty=1.5,
         )
         text = ""
         if stream:
            for message in chat_completion:
               text += message.choices[0].delta.content
         else:
            text = chat_completion.choices[0].message.content
         
         return text