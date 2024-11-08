import os
from dotenv import load_dotenv
import requests
from pydub import AudioSegment
import re

# Load environment variables
load_dotenv(".env")
MATCHA_URL = os.environ["MATCHA_URL"]
HF_TOKEN = os.environ["HF_TOKEN"]

# Headers for API requests
headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
}

# Dialogue text between Cai and LaIA
dialogue_text = """
Cai: Bon dia, LaIA. M'agradaria saber com funciona el procés de sol·licitud de la beca Equitat.

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

LaIA: De res, Cai. Si tens més preguntes, no dubtis a consultar-me.
""" 

# Extract and organize lines as tuples in the format (speaker, text)
lines = re.findall(r'(Cai|LaIA): (.+?)(?=\n(?:Cai|LaIA):|$)', dialogue_text, re.DOTALL)

# Function to generate TTS for a given speaker and text
def generate_audio(speaker, text, file_index):
    voice = "grau" if speaker == "Cai" else "elia"
    accent = "central"
    payload = {
        "text": text,
        "voice": voice,
        "accent": accent
    }
    
    try:
        response = requests.post(MATCHA_URL, headers=headers, json=payload)
        response.raise_for_status()
        audio_filename = f"{file_index}_{speaker}.wav"
        
        with open(audio_filename, "wb") as audio_file:
            audio_file.write(response.content)
        return audio_filename
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error: {http_err}")
    except Exception as err:
        print(f"Other error: {err}")
    return None

# Generate and store audio files for each line
audio_files = []
for i, (speaker, text) in enumerate(lines):
    print(f"Generating audio for: {speaker} - '{text[:50]}...'")  # Display the beginning of each line
    audio_file = generate_audio(speaker, text, i)
    if audio_file:
        audio_files.append(audio_file)

# Concatenate all generated audio files
final_audio = AudioSegment.empty()
for audio_file in audio_files:
    segment = AudioSegment.from_wav(audio_file)
    final_audio += segment

# Export final audio and delete intermediate files
final_audio.export("final_conversation.wav", format="wav")

# Clean up old individual audio files
for audio_file in audio_files:
    os.remove(audio_file)

print("Final audio saved as 'final_conversation.wav'.")