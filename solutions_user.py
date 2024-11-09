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



messages = [ {"role": "system", "content":  "Ets un assistent en català que ha de respondre únicament a la pregunta formulada per l'usuari, "
            "utilitzant de manera precisa la informació proporcionada. Les respostes han de ser clares, concises "
            "i directament relacionades amb la pregunta, sense afegir informació addicional o irrellevant."
            "No és necessari que la resposta sigui idèntica a la informació proporcionada del tràmit, però ha de ser coherent amb aquesta."
            "Si la informació proporcionada no és suficient per respondre amb precisió, "
            "indica a l'usuari que no disposes de la resposta amb la informació facilitada."}]

messages.append( {"role":"user", "content": """Respon de manera concisa a: Quin termini tinc per sol·licitar la llicència de caça per a un menor?"""})
messages.append( {"role":"user", "content": """A contiuació tens la informació del tràmit per tal de poder solucionar el dubte de l'usuari: """})
messages.append( {"role":"user", "content": """Llicència de caça
Què necessites fer?
Consulta a continuació totes les opcions vinculades a aquest tràmit. Selecciona la que correspongui amb el teu cas i podràs accedir a tota la informació i condicions de tramitació.
                  Sol·licitar la llicència
                  Què és?
La llicència de caça és el document nominal i intransferible obligatori per poder caçar dins del territori de Catalunya i pot tenir una vigència variable. Per obtenir-la cal pagar una taxa que posteriorment es reinverteix en la gestió, la conservació, la repoblació i la vigilància de la caça i dels hàbitats.
Per caçar, a més a més de la llicència, cal tenir el permís del titular del terreny en el cas de practicar la caça a les Àrees privades de caça; i cal tenir un permís específic per caçar a les Reserves nacionals de caça o a les Zones de caça controlada:

Hi ha diferents tipus de llicències de caça.
Llicència tipus A: Caça amb armes de foc i assimilables.
Llicència tipus B: Caça amb altres procediments autoritzats.
Llicència tipus C: Per tenir canilles.
Llicència tipus JA: Caça amb armes de foc per a majors de 65 anys.
Llicència tipus JB: Caça sense armes de foc per a majors de 65 anys.
Llicència tipus AT: Caça amb armes de foc i assimilables durant 15 dies seguits.

Segons la durada, hi ha llicències tipus A d'1,3 o 5 anys de vigència.

La llicència de caça no pot ser atorgada a persones que, per alguna raó, han estat inhabilitades per a la seva possessió. A tal efecte existeix un registre de persones inhabilitades. 
A qui va dirigit?
Ciutadania
A tothom que vulgui caçar al territori de Catalunya. Cal que la persona que ho sol·liciti sigui major de 14 anys, i els menors de 18 anys han de tenir una autorització del pare/mare/tutor.

Terminis
La  sol·licitud es pot presentar en qualsevol moment.

Documentació
Documents

Autorització a menors (Tipus de fitxer docx)(Pes del fitxer)[20 Bytes]

Taxes
Llicència tipus A: Caça amb armes de foc i assimilables: 29,40 €.

Llicència tipus B: Caça amb altres procediments autoritzats: 14,75 €.

Llicència tipus C: Per tenir canilles: 54,50 €.

Llicència tipus JA: Caça amb armes de foc per a majors de 65 anys: gratuïta

Llicència tipus JB: Caça sense armes de foc per a majors de 65 anys: gratuïta

Llicència tipus AT: Caça amb armes de foc i assimilables durant 15 dies seguits. 13,65 €

Llicència tipus A3: Caça amb armes de foc i assimilables durant 3 anys seguits: 79,60 €

Llicència tipus A5: Caça amb armes de foc i assimilables durant 5 anys seguits: 132,45 €

Altres informacions
Llicències de menors de 18 anys
En cas de sol·licitar una llicència de caça per a un menor de 18 anys, un cop sol·licitada cal accedir a l'apartat Aportar documentació, dins aquesta mateixa fitxa de tramitació. La documentació que cal aportar és la següent:

- Per a llicències de menors de 18 anys:

DNI/NIF/NIE o Passaport del pare o la mare o el/la tutor/a legal del menor. (Només en cas de s’aporti la documentació presencialment).
Autorització pel menor a obtenir la llicència, signada pel pare, la mare o el/la tutor/a legal.
Còpia de l'assegurança de responsabilitat civil de caçador del menor (no caldrà si en el formulari de sol·licitud consigneu les dades de la companyia asseguradora, el número de pòlissa i la data de caducitat de l’assegurança).
Per a àrees privades de caça: 
Cal tenir el permís del titular del terreny en el cas de caçar a les àrees privades de caça.

Per a reserves nacionals de caça o zones de caça controlada cal tenir un permís específic.
                  """})
stream = False
chat_completion = client.chat.completions.create(
   model="tgi",
   messages=messages,
   stream=stream,
   max_tokens=1000,
   temperature=0.075,
   top_p=0.95,
   frequency_penalty=0.05,
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