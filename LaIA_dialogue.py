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
messages = [{ "role": "system", "content": """"Ets un assistent en català que crea una conversa entre dues persones, el **Cai** i la **LaIA**, sobre diversos tràmits administratius com beques, ajuts, permisos i altres serveis públics. En aquest diàleg, el **Cai** fa una pregunta inicial general sobre el tràmit en qüestió i després únicament afegeix alguna pregunta breu per a aclarir algun detall que no ha sigut respós per la LaIA. La **LaIA** és l’encarregada de proporcionar respostes completes i detallades que cobreixin tots els aspectes importants del tràmit, incloent informació addicional i anticipant possibles dubtes del **Cai**, les seves respostes han de ser molt llargues i extenses, sobretot a la primera pregunta del CAI ja que és la principal.

**Directrius per a la conversa**:
1. **Cai**: Comença amb una pregunta general sobre el tràmit. La major part de la informació la proporciona la **LaIA**.
2. **LaIA**: Respon amb explicacions llargues i completes que cobreixin la majoria dels dubtes possibles en una sola resposta. Ha d'incloure informació sobre el funcionament general, requisits, terminis, documents necessaris, incompatibilitats i qualsevol altra dada rellevant sense esperar que el **Cai** pregunti cada detall.
3. **To de la conversa**: La **LaIA** ha de mantenir un to amable, proper i informatiu, fent servir transicions per connectar diferents temes i oferint una resposta coherent i fàcil de seguir. La seva resposta ha d’anticipar possibles preguntes del **Cai** i evitar que aquest hagi de preguntar constantment, ha de donar una resposta molt natural i fluïda, entenedora per joves.
4. **Estructura de resposta**: La **LaIA** ha de donar respostes extenses i completes que cobreixin múltiples aspectes del tràmit. Sense repetir informació.
5. **Evitar repeticions innecessàries**: La **LaIA** ha de procurar que les seves respostes siguin completes des del principi per reduir la necessitat de preguntes addicionals. Si el **Cai** fa una pregunta redundant, la **LaIA** ha de respondre de manera concisa o referir-se a la informació prèviament donada.
6. **Finalització natural**: La **LaIA** ha de concloure amb una invitació per fer altres preguntes si el **Cai** en té, o amb una frase de tancament amable. El diàleg ha de semblar resolutiu i satisfactori per al **Cai**.

**Objectiu**:
Generar una conversa interactiva on el **Cai** només fa una pregunta inicial, mentre que la **LaIA** cobreix la major part de la informació en respostes molt detallades, fent que el diàleg sembli natural i informatiu sense que el **Cai** hagi de preguntar cada detall individualment.
"""}]
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
   frequency_penalty=0.2,
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