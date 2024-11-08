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
messages = [{ "role": "system", "content": "Ets un assistent útil."}]
messages.append( {"role":"user", "content": """Resumeix molt breument i tradueix el següent text a català: EXPOSICION DE MOTIVOS
Si se ha llegado a definir el ordenamiento jurídico como conjunto de normas que regulan el uso de la fuerza, puede entenderse fácilmente la importancia del Código Penal en cualquier sociedad civilizada. El Código Penal define los delitos y faltas que constituyen los presupuestos de la aplicación de la forma suprema que puede revestir el poder coactivo del Estado: la pena criminal. En consecuencia, ocupa un lugar preeminente en el conjunto del ordenamiento, hasta el punto de que, no sin razón, se ha considerado como una especie de «Constitución negativa». El Código Penal ha de tutelar los valores y principios básicos de la convivencia social. Cuando esos valores y principios cambian, debe también cambiar. En nuestro país, sin embargo, pese a las profundas modificaciones de orden social, económico y político, el texto vigente data, en lo que pudiera considerarse su núcleo básico, del pasado siglo. La necesidad de su reforma no puede, pues, discutirse.

A partir de los distintos intentos de reforma llevados a cabo desde la instauración del régimen democrático, el Gobierno ha elaborado el proyecto que somete a la discusión y aprobación de las Cámaras. Debe, por ello, exponer, siquiera sea de modo sucinto, los criterios en que se inspira, aunque éstos puedan deducirse con facilidad de la lectura de su texto.

El eje de dichos criterios ha sido, como es lógico, el de la adaptación positiva del nuevo Código Penal a los valores constitucionales. Los cambios que introduce en esa dirección el presente proyecto son innumerables, pero merece la pena destacar algunos.

En primer lugar, se propone una reforma total del actual sistema de penas, de modo que permita alcanzar, en lo posible, los objetivos de resocialización que la Constitución le asigna. El sistema que se propone simplifica, de una parte, la regulación de las penas privativas de libertad, ampliando, a la vez, las posibilidades de sustituirlas por otras que afecten a bienes jurídicos menos básicos, y, de otra, introduce cambios en las penas pecuniarias, adoptando el sistema de días-multa y añade los trabajos en beneficio de la comunidad.

En segundo lugar, se ha afrontado la antinomia existente entre el principio de intervención mínima y las crecientes necesidades de tutela en una sociedad cada vez más compleja, dando prudente acogida a nuevas formas de delincuencia, pero eliminando, a la vez, figuras delictivas que han perdido su razón de ser. En el primer sentido, merece destacarse la introducción de los delitos contra el orden socioeconómico o la nueva regulación de los delitos relativos a la ordenación del territorio y de los recursos naturales; en el segundo, la desaparición de las figuras complejas de robo con violencia e intimidación en las personas que, surgidas en el marco de la lucha contra el bandolerismo, deben desaparecer dejando paso a la aplicación de las reglas generales.

En tercer lugar, se ha dado especial relieve a la tutela de los derechos fundamentales y se ha procurado diseñar con especial mesura el recurso al instrumento punitivo allí donde está en juego el ejercicio de cualquiera de ellos: sirva de ejemplo, de una parte, la tutela específica de la integridad moral y, de otra, la nueva regulación de los delitos contra el honor. Al tutelar específicamente la integridad moral, se otorga al ciudadano una protección más fuerte frente a la tortura, y al configurar los delitos contra el honor del modo en que se propone, se otorga a la libertad de expresión toda la relevancia que puede y debe reconocerle un régimen democrático.

En cuarto lugar, y en consonancia con el objetivo de tutela y respeto a los derechos fundamentales, se ha eliminado el régimen de privilegio de que hasta ahora han venido gozando las injerencias ilegítimas de los funcionarios públicos en el ámbito de los derechos y libertades de los ciudadanos. Por tanto, se propone que las detenciones, entradas y registros en el domicilio llevadas a cabo por autoridad o funcionario fuera de los casos permitidos por la Ley, sean tratadas como formas agravadas de los correspondientes delitos comunes, y no como hasta ahora lo han venido siendo, esto es, como delitos especiales incomprensible e injustificadamente atenuados."""})
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