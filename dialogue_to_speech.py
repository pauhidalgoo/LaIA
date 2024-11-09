import os
import re
import ssl
import certifi
import asyncio
import aiohttp
from dotenv import load_dotenv
from pydub import AudioSegment

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

# Async function to generate TTS for a given speaker and text
async def generate_audio(session, speaker, text, file_index):
    voice = "grau" if speaker == "Cai" else "elia"
    accent = "central"
    payload = {
        "text": text,
        "voice": voice,
        "accent": accent
    }
    audio_filename = f"{file_index}_{speaker}.wav"
    
    try:
        async with session.post(MATCHA_URL, headers=headers, json=payload) as response:
            response.raise_for_status()
            with open(audio_filename, "wb") as audio_file:
                audio_file.write(await response.read())
        return audio_filename
    except aiohttp.ClientError as e:
        print(f"Client error: {e}")
    except Exception as err:
        print(f"Error: {err}")
    return None

# Main async function to process all lines and generate audio files
async def process_audio_files():
    ssl_context = ssl.create_default_context(cafile=certifi.where())  # Use certifi for SSL context
    connector = aiohttp.TCPConnector(ssl=ssl_context)  # Set up connector with SSL context

    time_stamps = []  # List to store (start, end, text) for each line
    start_time = 0    # Initialize start time for the first line

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            generate_audio(session, speaker, text, i)
            for i, (speaker, text) in enumerate(lines)
        ]
        
        # Gather all audio file names
        audio_files = await asyncio.gather(*tasks)
        audio_files = [file for file in audio_files if file is not None]  # Filter out None values

        # Concatenate all generated audio files
        final_audio = AudioSegment.empty()
        for i, audio_file in enumerate(audio_files):
            segment = AudioSegment.from_wav(audio_file)
            end_time = start_time + segment.duration_seconds  # Calculate end time for the line
            time_stamps.append((start_time, end_time, lines[i][1]))  # Append (start, end, text) to the list
            start_time = end_time  # Update start time for the next line
            final_audio += segment

        # Speed up final audio by 1.15x
        final_audio = final_audio.speedup(playback_speed=1.15)
        
        # Export final audio and delete intermediate files
        final_audio.export("final_conversation.wav", format="wav")
        for audio_file in audio_files:
            os.remove(audio_file)
        print("Final audio saved as 'final_conversation.wav'.")
    
    # Output the time stamps
    print("Time Stamps for Each Line:")
    for start, end, text in time_stamps:
        print(f"Start: {start:.2f}s, End: {end:.2f}s, Text: {text}")

# Run the async process
asyncio.run(process_audio_files())