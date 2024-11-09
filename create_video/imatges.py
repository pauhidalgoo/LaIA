import os
import re
from dotenv import load_dotenv
from openai import OpenAI
import random
load_dotenv(".env")

HF_TOKEN = os.environ["HF_TOKEN"]
BASE_URL = os.environ["BASE_URL"]


#pip install openai
client = OpenAI(
       base_url=BASE_URL + "/v1/",
       api_key=HF_TOKEN
   )

# Function to retrieve complete image paths within the 'images' folder
# Function to retrieve folder names within the 'images' directory
def get_image_categories(base_path):
    categories = []
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            categories.append(folder_name)
    return categories

# Format folder names as a readable list for the model prompt
def format_categories_for_model(categories):
    categories_text = '\n'.join(categories)
    return categories_text

# Function to get all images from the selected category folder
def extract_selected_images(base_path, selected_category):
    selected_images = []
    category_path = os.path.join(base_path, selected_category)
    for file_name in os.listdir(category_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Add more extensions if needed
            selected_images.append(os.path.join(category_path, file_name))
    return selected_images

# Usage example
base_path = "images"  # Define the path where images are located
categories = get_image_categories(base_path)
categories_text = format_categories_for_model(categories)

# Create a prompt for the model, including the list of image paths
prompt = f"""
Here is a dialogue between two individuals about an administrative procedure.

Please choose the most relevant category for the dialogue topic from the following options, each representing a set of background images:

Categories:
{categories_text}

Once you have selected the category, I will retrieve all images within that folder.
"""

messages = [{ "role": "system", "content": f'{prompt}'}]

messages.append( {"role":"user", "content": """A continuació tens el diàleg: """})
messages.append( {"role":"user", "content": """Cai: Bon dia, LaIA. M'agradaria saber com funciona el procés de sol·licitud de la beca Equitat.

LaIA: Bon dia, Cai. La beca Equitat és una ajuda econòmica que s'ofereix a estudiants de grau i màster habilitant de les universitats públiques de Catalunya, la UOC i alguns centres adscrits.

Cai: Com funciona el procés de sol·licitud?

LaIA: El procés de sol·licitud es realitza exclusivament per internet i el termini per al curs 2024-2025 és del 16 de setembre al 31 d'octubre de 2024 a les 14:00 h (hora local de Barcelona).

Cai: Quins requisits he de complir per a sol·licitar la beca?

LaIA: Per a sol·licitar la beca Equitat, cal que siguis un estudiant de grau o màster habilitant de les universitats públiques de Catalunya, la UOC o alguns centres adscrits que hi participen.

Cai: Com sé si puc optar a la beca?

LaIA: Per a saber si pots optar a la beca Equitat, has de demanar la beca i, posteriorment, l'AGAUR t'informarà de quin tram de renda familiar et correspon.

Cai: Com es demana la beca?

LaIA: La sol·licitud es realitza per via electrònica i has d'accedir a través de l'apartat "Tràmits gencat" del web de la Generalitat de Catalunya o des de la pàgina web de l'AGAUR.

Cai: Com sé l'estat de la meva beca?

LaIA: La tramitació de les sol·licituds de beques gestionades per l'AGAUR es realitza a través del portal Tràmits gencat, amb el codi identificador (codi ID) associat a la sol·licitud.

Cai: Què és el codi ID?

LaIA: El codi ID és un codi identificador associat a la sol·licitud de la beca i consta en el resguard de la beca sol·licitada.

Cai: Com sé si la beca cobreix el preu de la matrícula?

LaIA: La beca Equitat cobreix la minoració del preu de la matrícula, però no cobreix els crèdits matriculats per segona i successives vegades, els crèdits convalidats, reconeguts i/o adaptats i les quotes de l'assegurança escolar o qualsevol altra associada a la matrícula.

Cai: És compatible amb la beca general del Ministeri?

LaIA: La beca Equitat és incompatible amb la beca general del Ministeri.

Cai: És recomanable demanar les dues beques?

LaIA: Sí, és recomanable demanar les dues beques, ja que la beca Equitat complementa la beca general del Ministeri.

Cai: Moltes gràcies, LaIA.

LaIA: De res, Cai. Si tens més preguntes, no dubtis a consultar-me."""})

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
else:
 text = chat_completion.choices[0].message.content

# Function to detect the selected_category from the model output, the class are stored in categories variable
def detect_selected_category(text, categories):
    for category in categories:
        if category.lower() in text.lower():
            return category
    return random.choice(categories)

# Extract selected images from model output
def extract_selected_images(base_path, selected_category):
    selected_images = []
    category_path = os.path.join(base_path, selected_category)
    for file_name in os.listdir(category_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Add more extensions if needed
            selected_images.append(os.path.join(category_path, file_name))
    return selected_images

selected_images = extract_selected_images("images", detect_selected_category(text, categories))

# Delete all images from generated_images folder
for file in os.listdir("generated_images"):
    file_path = os.path.join("generated_images", file)
    if os.path.isfile(file_path):
        os.unlink(file_path)

# Copy selected images to generated_images folder without shutil
for image in selected_images:
    image_name = os.path.basename(image)
    new_image_path = os.path.join("generated_images", image_name)
    with open(image, "rb") as f:
        with open(new_image_path, "wb") as new_f:
            new_f.write(f.read())