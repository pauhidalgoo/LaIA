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
messages = [{ "role": "system", "content": """"Ets un assistent en català que crea un diàleg entre dues persones, el Cai i la LaIA, sobre diversos tràmits administratius com beques, ajuts, permisos i altres serveis públics. En aquest diàleg, el Cai fa una pregunta inicial general sobre el tràmit i només afegeix alguna intervenció molt breu per mantenir la conversa natural, sense entrar en detalls. La LaIA és qui proporciona respostes completes, cobertes de tots els aspectes importants, incloent-hi informació addicional i anticipant qualsevol dubte que el Cai pugui tenir.

Directrius per a la conversa:

Què fa el personatge Cai: Comença amb una pregunta general sobre el tràmit i fa molt poques intervencions breus després, només per mantenir la conversa fluida. Evita fer preguntes detallades o segmentades; el seu paper és principalment reactiu.

Què fa el personatge LaIA: Respon amb explicacions llargues, completes i naturals, cobrint tots els aspectes possibles del tràmit en una sola resposta extensa. Ha de proporcionar informació sobre el funcionament, requisits, terminis, documents necessaris, possibles incompatibilitats i consells, de manera que el Cai no hagi de preguntar més detalls. La LaIA s’assegura d'anticipar qualsevol dubte que podria sorgir.

Com ha de ser el to de la conversa: La LaIA ha de sonar propera, informativa i accessible. La seva resposta ha d’incloure detalls i transicions suaus entre temes, oferint una explicació fàcil de seguir i completa, com si parlés amb algú en una conversa natural. La LaIA i el Cai han de mantenir un to amable i les seves interaccions han d'estar connectades de manera coherent.

Com és l'estructura de resposta: La LaIA cobreix tots els temes rellevants en una resposta completa, fent que el Cai no senti la necessitat de fer preguntes addicionals. Si el Cai fa una intervenció breu de seguiment, la LaIA pot respondre de manera concisa o fer una referència breu a la informació donada.

Quanta informació ha d'incloure cada resposta: La LaIA ha de donar respostes inicials molt completes per minimitzar la necessitat de preguntes de seguiment. Si el Cai fa una pregunta redundant, la LaIA ha de mantenir la resposta breu i referir-se a la informació prèvia.

Com ha de finalitzar el diàleg: La LaIA ha de tancar amb una frase amable, indicant que pot ajudar en cas de dubtes, però deixant clar que ja ha proporcionat tota la informació necessària perquè el Cai se senti ben informat.

Quin és l'objectiu final: Generar una conversa on el Cai només fa una pregunta inicial general i, com a màxim, alguna intervenció molt breu i reactiva. La LaIA cobreix tota la informació en una resposta detallada i anticipativa, fent que el diàleg sembli natural, complet i informatiu sense necessitat de més preguntes per part del Cai.
             
S'ha de complir: La LaIA respon molt bé les preguntes incloent tota la informació possible i encara més detalls extres així que el Cai només ha de preguntar com a molt 3 vegades qüestions molt específiques. S'ha de complir que les preguntes del Cai comencin de maneres diferents per evitar repeticions quan el Cai parla ha de donar gràcies a la Laia o dir-li d'acord d'alguna manera. Quan la Laia parla, si Cai li ha fet una molt bona pregunta cal que li comenti que és molt important el que ha dit fent molt més natural la interacció.
"""}]

messages.append( {"role":"user", "content": """A contiuació tens la informació del tràmit per tal de poder crear el diàleg: """})
messages.append( {"role":"user", "content": """IMPORTANT:

Tots els tràmits d'aquesta convocatòria es faran exclusivament per internet.

La beca Equitat consisteix en l’aplicació d’un percentatge de minoració fins a un màxim del 80%, segons el tram de renda assignat i els estudis que cursis, en el preu de la matrícula (crèdits ordinaris matriculats per 1a vegada).

L’acreditació de tram de renda familiar és un document que us informarà de quin tram de renda familiar us correspon (entre l'1 i el 2 o fora de trams). Aquest document és exclusivament de caràcter informatiu i no genera el dret a la beca Equitat.

No cal demanar la beca general per optar-hi.

Qui pot sol·licitar la beca?

Persones que cursin estudis universitaris oficials de grau o de màster habilitant en una de les universitats públiques de Catalunya, a la UOC o a algun dels 3 centres adscrits que hi participen.

Quan la sol·licito?

Termini de sol·licituds per al curs 2024-2025: Del 16 de setembre al 31 d'octubre de 2024 a les 14:00 h (hora local de Barcelona) ambdós inclosos.

La sol·licitud es realitza per via electrònica. Heu d'accedir a través de l'apartat "Tràmits gencat" del web de la Generalitat de Catalunya o des de la pàgina web de l'AGAUR.

Cal disposar d'un mecanisme d'identificació digital, a nom de la persona sol·licitant. Us recomanem l'idCat Mòbil.

Com sabré l’estat de la meva beca? Què és el codi ID?

La tramitació de les sol·licituds de beques gestionades per l'AGAUR es realitza a través del portal Tràmits gencat, amb el codi identificador (codi ID) associat a la sol·licitud.

Aquest codi ID consta en el resguard de la beca sol·licitada.

Per a més informació, consulta les preguntes més freqüents.

Els ajuts depenen del nombre de crèdits matriculats? Quin import cobreix la beca de matrícula?

No hi ha requisits acadèmics.

No obstant la minoració del preu de la matrícula no cobreix els següents conceptes: Els preus de gestió de la matrícula, de suport a l’aprenentatge, dels crèdits matriculats per segona i successives vegades, els crèdits convalidats, reconeguts i/o adaptats i les quotes de l’assegurança escolar o qualsevol altra associada a la matrícula.

Tampoc els crèdits que excedeixin del mínim necessari per a obtenir la titulació per a la qual se sol·licita la beca.

És incompatible amb la beca general del Ministeri?

Si se’t concedeix la beca General, la beca Equitat quedarà sense efecte. En cas que se’t denegui la beca General, si has demanat també la beca equitat i se't concedeix, se t'aplicarà la minoració que et correspongui.

És recomanable que les demanis totes dues beques."""})
stream = False
chat_completion = client.chat.completions.create(
   model="tgi",
   messages=messages,
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
   print(message.choices[0].delta.content, end="")
 print(text)
else:
 text = chat_completion.choices[0].message.content
 print(text)