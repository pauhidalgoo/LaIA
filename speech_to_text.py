import requests
import os
import sounddevice as sd
import numpy as np
import wave
from dotenv import load_dotenv
import logging
from pathlib import Path
import time
from tqdm import tqdm

class SpeechToText:
	def __init__(self, whisper_url: str, hf_token: str):
		"""Initialize the Speech-to-Text service with configuration parameters."""
		self.whisper_url = whisper_url
		self.headers = {
			"Accept": "application/json",
			"Authorization": f"Bearer {hf_token}",
			"Content-Type": "audio/wav",
		}
		
		# Audio configuration
		self.sample_rate = 16000
		self.channels = 1
		self.silence_threshold = 0.03  # Adjusted threshold for better silence detection
		self.chunk_duration = 1024
		self.silence_duration = 2.0  # Seconds of silence before stopping
		self.max_duration = 30.0  # Maximum recording duration in seconds
		
		# Setup logging
		logging.basicConfig(level=logging.INFO)
		self.logger = logging.getLogger(__name__)
		
		# Recording state
		self.recording = []
		self.silent_chunks = 0
		self.is_recording = False

	def start_recording(self, filename="output.wav"):
		"""Start recording audio with improved error handling and feedback."""
		try:
			# Convert Path to string if it's a Path object
			filename = str(filename) if isinstance(filename, Path) else filename
			
			# Ensure directory exists
			os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
			
			self.recording = []
			self.is_recording = True
			self.silent_chunks = 0
			
			# Configure sounddevice
			sd.default.samplerate = self.sample_rate
			sd.default.channels = self.channels
			
			# Calculate parameters for silence detection
			self.chunks_per_second = self.sample_rate / self.chunk_duration
			self.silence_chunks = int(self.silence_duration * self.chunks_per_second)
			
			with sd.InputStream(
			   	callback=self.__audio_callback,
			   	channels=self.channels,
			   	samplerate=self.sample_rate,
			   	blocksize=self.chunk_duration
			):
				self.logger.info("Recording started... Speak now")
				progress_bar = tqdm(total=self.max_duration, desc="Recording", unit="s")
				start_time = time.time()

				while self.is_recording and len(self.recording) < self.max_duration * self.chunks_per_second:
					sd.sleep(100)
					elapsed_time = time.time() - start_time
					progress_bar.update(elapsed_time - progress_bar.n)

				progress_bar.close()
				
			return self.__save_recording(filename)
			
		except Exception as e:
			self.logger.error(f"Error during recording: {str(e)}")
			raise

	def __audio_callback(self, indata, frames, time, status):
		"""Process audio input with improved silence detection."""
		if status:
			self.logger.warning(f"Status: {status}")
		
		# Calculate RMS amplitude
		amplitude = np.sqrt(np.mean(indata**2))
		
		if amplitude < self.silence_threshold:
			self.silent_chunks += 1
			if self.silent_chunks >= self.silence_chunks:
				self.is_recording = False
		else:
			self.silent_chunks = 0
			
		if self.is_recording:
			self.recording.append(indata.copy())

	def __save_recording(self, filename):
		"""Save the recording with proper WAV formatting and verification."""
		if not self.recording:
			self.logger.error("No audio data recorded")
			return None

		try:
			# Combine all chunks
			audio_data = np.concatenate(self.recording, axis=0)
			
			# Convert to 16-bit PCM
			audio_data = (audio_data * np.iinfo(np.int16).max).astype(np.int16)
			
			# Ensure filename is string, not Path
			filename = str(filename)
			
			with wave.open(filename, 'wb') as wf:
				wf.setnchannels(self.channels)
				wf.setsampwidth(2)  # 16 bits per sample
				wf.setframerate(self.sample_rate)
				wf.writeframes(audio_data.tobytes())
			
			# Verify file was created and has content
			if os.path.exists(filename) and os.path.getsize(filename) > 0:
				self.logger.info(f"Recording saved successfully to {filename}")
				return filename
			else:
				raise Exception("Failed to save recording: File is empty or not created")
				
		except Exception as e:
			self.logger.error(f"Error saving recording: {str(e)}")
			raise

	def transcribe_audio(self, filename: str):
		"""Transcribe audio with error handling and response validation."""
		try:
			# Convert Path to string if it's a Path object
			filename = str(filename) if isinstance(filename, Path) else filename
			
			if not os.path.exists(filename):
				raise FileNotFoundError(f"Audio file not found: {filename}")
				
			with open(filename, "rb") as f:
				data = f.read()
				
			if len(data) == 0:
				raise ValueError("Audio file is empty")
				
			response = requests.post(
				self.whisper_url,
				headers=self.headers,
				data=data,
				timeout=30
			)
			
			response.raise_for_status()
			result = response.json()
			
			if "error" in result:
				raise Exception(f"Transcription API error: {result['error']}")
				
			return result
			
		except requests.exceptions.RequestException as e:
			self.logger.error(f"API request failed: {str(e)}")
			raise
		except Exception as e:
			self.logger.error(f"Transcription error: {str(e)}")
			raise

def main():
	"""Main function with environment validation and error handling."""
	try:
		load_dotenv(".env")
		whisper_url = os.environ["WHISPER_URL"]
		hf_token = os.environ["HF_TOKEN"]
		
		if not whisper_url or not hf_token:
			raise ValueError("Missing required environment variables: WHISPER_URL or HF_TOKEN")
			
		stt_service = SpeechToText(whisper_url, hf_token)
		
		# Create output directory
		output_dir = Path("./data/audio")
		output_dir.mkdir(parents=True, exist_ok=True)
		
		# Convert Path to string for the filename
		output_file = str(output_dir / "output.wav")
		
		filename = stt_service.start_recording(output_file)
		if filename:
			transcription = stt_service.transcribe_audio(filename)
			print("Transcription result:", transcription)
	
	except Exception as e:
		logging.error(f"Application error: {str(e)}")
		raise

if __name__ == "__main__":
	main()