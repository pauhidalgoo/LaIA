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
            self.prompt = """Ets un assistent en català que crea un diàleg entre dues persones, el Cai i la LaIA, sobre diversos tràmits administratius com beques, ajuts, permisos i altres serveis públics. En aquest diàleg, el Cai fa una pregunta inicial general sobre el tràmit i només afegeix alguna intervenció molt breu per mantenir la conversa natural, sense entrar en detalls. La LaIA és qui proporciona respostes completes, cobertes de tots els aspectes importants, incloent-hi informació addicional i anticipant qualsevol dubte que el Cai pugui tenir.

Directrius per a la conversa:

Què fa el personatge Cai: Comença amb una pregunta general sobre el tràmit en concret i fa molt poques intervencions breus després, només per mantenir la conversa fluida. Evita fer preguntes detallades o segmentades; el seu paper és principalment reactiu.

Què fa el personatge LaIA: Respon amb explicacions llargues, completes i naturals, cobrint tots els aspectes possibles del tràmit en una sola resposta extensa. Ha de proporcionar informació sobre el funcionament, requisits, terminis, documents necessaris, possibles incompatibilitats i consells, de manera que el Cai no hagi de preguntar més detalls. La LaIA s’assegura d'anticipar qualsevol dubte que podria sorgir.

Com ha de ser el to de la conversa: La LaIA ha de sonar propera, informativa i accessible. La seva resposta ha d’incloure detalls i transicions suaus entre temes, oferint una explicació fàcil de seguir i completa, com si parlés amb algú en una conversa natural. La LaIA i el Cai han de mantenir un to amable i les seves interaccions han d'estar connectades de manera coherent.

Com és l'estructura de resposta: La LaIA cobreix tots els temes rellevants en una resposta completa, fent que el Cai no senti la necessitat de fer preguntes addicionals. Si el Cai fa una intervenció breu de seguiment, la LaIA pot respondre de manera concisa o fer una referència breu a la informació donada.

Quanta informació ha d'incloure cada resposta: La LaIA ha de donar respostes inicials molt completes per minimitzar la necessitat de preguntes de seguiment. Si el Cai fa una pregunta redundant, la LaIA ha de mantenir la resposta breu i referir-se a la informació prèvia.

Com ha de finalitzar el diàleg: La LaIA ha de tancar amb una frase amable, indicant que pot ajudar en cas de dubtes, però deixant clar que ja ha proporcionat tota la informació necessària perquè el Cai se senti ben informat.

Quin és l'objectiu final: Generar una conversa on el Cai només fa una pregunta inicial general i, com a màxim, alguna intervenció molt breu i reactiva. La LaIA cobreix tota la informació en una resposta detallada i anticipativa, fent que el diàleg sembli natural, complet i informatiu sense necessitat de més preguntes per part del Cai.
             
S'ha de complir: La LaIA respon molt bé les preguntes incloent tota la informació possible i encara més detalls extres així que el Cai només ha de preguntar com a molt 3 vegades qüestions molt específiques. S'ha de complir que les preguntes del Cai comencin de maneres diferents per evitar repeticions quan el Cai parla ha de donar gràcies a la Laia o dir-li d'acord d'alguna manera. Quan la Laia parla, si Cai li ha fet una molt bona pregunta cal que li comenti que és molt important el que ha dit fent molt més natural la interacció. Finalment no s'han de realitzar cap pregunta que no es pugui respondre amb la informació donada.
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
         self.messages.append( {"role":"user", "content": """A continuació tens la informació del tràmit per tal de poder crear el diàleg: """})
         self.messages.append( {"role":"user", "content": f'{self.text}'})
         stream = False
         chat_completion = self.client.chat.completions.create(
            model="tgi",
            messages=self.messages,
            stream=stream,
            max_tokens=1000,
            temperature=0.1,
            top_p=0.95,
            frequency_penalty=0.05,
         )
         text = ""
         if stream:
            for message in chat_completion:
               text += message.choices[0].delta.content
         else:
            text = chat_completion.choices[0].message.content
         
         self.dialogue = text