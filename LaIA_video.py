import os
import re
import ssl
import certifi
import asyncio
import aiohttp
from dotenv import load_dotenv
from pydub import AudioSegment
from moviepy.editor import ImageClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
from textwrap import fill
from openai import OpenAI
import random


class LaIA_video:
    def __init__(self, dialogue_text='', dialect='central', voices = {'man':'grau', 'woman':'elia'}):
        assert dialogue_text, "Text is required"
        assert dialect in ["central", "nord-occidental", "balear", "valencia"], "Invalid dialect"

        self.dialogue_text = dialogue_text
        self.dialect = dialect
        self.voices = voices

        # Load environment variables
        load_dotenv(".env")
        self.BASE_URL = os.environ["BASE_URL"]
        self.API_URL = os.environ["API_URL"]
        self.MATCHA_URL = os.environ["MATCHA_URL"]
        self.HF_TOKEN = os.environ["HF_TOKEN"]
        self._client()

        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.HF_TOKEN}",
        }

        # Create audio
        self.create_audio()

        # Create video
        self.create_video()

    def _client(self):
        """
        Create an OpenAI client
        """
        self.client = OpenAI(
            base_url=f"{self.BASE_URL}/v1/",
            api_key=self.HF_TOKEN
        )
    
    def create_audio(self):
        # Extract and organize lines as tuples in the format (speaker, text)
        self.lines = re.findall(r'(Cai|LaIA): (.+?)(?=\n(?:Cai|LaIA):|$)', self.dialogue_text, re.DOTALL)

        # Async function to generate TTS for a given speaker and text
        async def generate_audio(session, speaker, text, file_index):
            voice = self.voices['man'] if speaker == "Cai" else self.voices['woman']
            accent = self.dialect
            payload = {"text": text, "voice": voice, "accent": accent}
            audio_filename = f"{file_index}_{speaker}.wav"
            
            try:
                async with session.post(self.MATCHA_URL, headers=self.headers, json=payload) as response:
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
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_context)  # Set up connector with SSL context

            async with aiohttp.ClientSession(connector=connector) as session:
                tasks = [generate_audio(session, speaker, text, i) for i, (speaker, text) in enumerate(self.lines)]
                
                # Gather all audio file names
                audio_files = await asyncio.gather(*tasks)
                audio_files = [file for file in audio_files if file is not None]  # Filter out None values

                # Concatenate all generated audio files
                final_audio = AudioSegment.empty()
                time_stamps = []
                start_time = 0

                for i, audio_file in enumerate(audio_files):
                    segment = AudioSegment.from_wav(audio_file)
                    end_time = start_time + segment.duration_seconds / 1.15  # Adjust end time with speed-up factor
                    time_stamps.append((start_time, end_time, self.lines[i][1]))  # Store adjusted timestamps
                    start_time = end_time
                    final_audio += segment

                # Speed up final audio by 1.15x
                final_audio = final_audio.speedup(playback_speed=1.15)
                
                # Export final audio and delete intermediate files
                final_audio.export("final_conversation.wav", format="wav")
                for audio_file in audio_files:
                    os.remove(audio_file)
                print("Final audio saved as 'final_conversation.wav'.")
            
            return time_stamps
        
        # Run the async process
        self.time_stamps = asyncio.run(process_audio_files())

    def generate_images(self):
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
        messages.append( {"role":"user", "content": f'{self.dialogue_text}'})

        stream = False
        chat_completion = self.client.chat.completions.create(
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

    def create_video(self):
        # Now create the video with subtitles using the generated audio and images
        image_folder = "generated_images"
        audio_file = "final_conversation.wav"
        output_video = "final_video_with_subtitles.mp4"
        image_duration = 10

        # Load audio
        audio = AudioFileClip(audio_file)
        audio_duration = audio.duration

        self.generate_images()

        # Get list of image files and check if there are any images
        image_files = sorted([os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")])

        if len(image_files) == 0:
            raise ValueError("No images found in the 'generated_images' folder.")
        
        # Create video clips for each image and repeat them to cover the full audio duration
        image_clips = [ImageClip(image_files[i % len(image_files)]).set_duration(image_duration).resize(height=720) for i in range(0, int(audio_duration // image_duration) + 1)]
        background_video = concatenate_videoclips(image_clips, method="compose").set_duration(audio_duration)

        # Create video clips with text for each segment in time_stamps
        text_clips = []
        max_text_width = 40

        for start, end, text in self.time_stamps:
            formatted_text = fill(text, width=max_text_width)
            text_clip = TextClip(formatted_text, fontsize=24, color='white', font="Montserrat-Bold", bg_color='rgba(0, 0, 0, 0.5)', size=(background_video.w, None), method="caption")
            text_clip = text_clip.set_duration(end - start).set_position(("center", "center")).set_start(start)
            text_clips.append(text_clip)

        # Combine background video and text clips
        final_video = CompositeVideoClip([background_video] + text_clips).set_audio(audio).set_fps(24)

        # Delete all images from generated_images folder
        for file in os.listdir(image_folder):
            file_path = os.path.join(image_folder, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

        # Export final video
        final_video.write_videofile(output_video, codec="libx264", audio_codec="aac", temp_audiofile="temp-audio.m4a", remove_temp=True, verbose=False)

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

# Example

LaIA_video(dialogue_text=dialogue_text, dialect='central')