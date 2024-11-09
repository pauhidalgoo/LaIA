import requests
import os
import sounddevice as sd
import numpy as np
import wave
from dotenv import load_dotenv
import logging
from pathlib import Path
import time
import json
import queue
import threading
import io


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
		self.silence_threshold = 0.05
		self.chunk_duration = 1024
		self.silence_duration = 2.0
		self.max_duration = 30.0
		
		# Setup logging
		logging.basicConfig(level=logging.INFO)
		self.logger = logging.getLogger(__name__)
		
		# Recording and streaming state
		self.recording = []
		self.silent_chunks = 0
		self.is_recording = False
		self.audio_queue = queue.Queue()
		self.transcription_thread = None
		self.current_transcription = ""

		self.audio_buffer = {}
		self.transcription_threads = {}

	def calibrate_silence_threshold(self, duration=2):
		"""Calibrate silence threshold based on ambient noise."""
		self.logger.info("Calibrating silence threshold...")
		silence_samples = []
		
		with sd.InputStream(
			channels=self.channels,
			samplerate=self.sample_rate,
			blocksize=self.chunk_duration
		) as stream:
			start_time = time.time()
			while time.time() - start_time < duration:
				indata, _ = stream.read(self.chunk_duration)
				amplitude = np.sqrt(np.mean(indata**2))
				silence_samples.append(amplitude)
				sd.sleep(100)
		
		# Set silence threshold based on average ambient noise
		self.silence_threshold = np.mean(silence_samples) * 1.5
		self.logger.info(f"Calibrated silence threshold to: {self.silence_threshold}")
	
	def stop_recording(self):
		"""Forcefully stop the recording."""
		self.is_recording = False
		self.logger.info("Recording forcefully stopped.")

	def start_recording(self, filename="output.wav"):
		"""Start recording audio and transcribing in real-time."""
		try:
			filename = str(filename) if isinstance(filename, Path) else filename
			os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

			self.recording = []
			self.is_recording = True
			self.silent_chunks = 0
			self.current_transcription = ""

			# Configure sounddevice
			sd.default.samplerate = self.sample_rate
			sd.default.channels = self.channels

			# Calibrate silence threshold dynamically
			self.calibrate_silence_threshold(duration=2)

			# Calculate parameters
			self.chunks_per_second = self.sample_rate / self.chunk_duration
			self.silence_chunks = int(self.silence_duration * self.chunks_per_second)

			# Start transcription thread
			self.transcription_thread = threading.Thread(target=self.__transcribe_stream)
			self.transcription_thread.start()

			with sd.InputStream(
				callback=self.__audio_callback,
				channels=self.channels,
				samplerate=self.sample_rate,
				blocksize=self.chunk_duration
			):
				self.logger.info("Recording started... Speak now")
				start_time = time.time()

				while self.is_recording:
					sd.sleep(100)
					
					# Print current transcription in real-time
					if self.current_transcription:
						print(f"\rTranscription: {self.current_transcription}", end="", flush=True)

				print("\n")  # New line after recording ends

			# Signal transcription thread to stop and wait for it
			self.audio_queue.put(None)
			self.transcription_thread.join()

			return self.__save_recording(filename)

		except Exception as e:
			self.logger.error(f"Error during recording: {str(e)}")
			raise

	def __audio_callback(self, indata, frames, time, status):
		"""Process audio input and add to queue for streaming transcription."""
		if status:
			self.logger.warning(f"Status: {status}")

		amplitude = np.sqrt(np.mean(indata**2))

		# Detect silence and stop recording if silence exceeds threshold
		if amplitude < self.silence_threshold:
			self.silent_chunks += 1
			if self.silent_chunks >= self.silence_chunks:
				self.is_recording = False
		else:
			self.silent_chunks = 0  # Reset silence count if sound is detected

		# Add audio chunk to queue for streaming transcription
		self.recording.append(indata.copy())
		self.audio_queue.put(indata.copy())

	def __transcribe_stream(self):
		"""Continuously transcribe audio chunks from the queue."""
		accumulated_audio = []
		
		while True:
			chunk = self.audio_queue.get()
			if chunk is None:  # Stop signal
				break
				
			accumulated_audio.append(chunk)
			
			# Process accumulated audio every ~2 seconds
			if len(accumulated_audio) >= int(2 * self.chunks_per_second):
				try:
					# Combine chunks and convert to WAV format
					audio_data = np.concatenate(accumulated_audio, axis=0)
					audio_data = (audio_data * np.iinfo(np.int16).max).astype(np.int16)
					
					# Create in-memory WAV file
					import io
					wav_buffer = io.BytesIO()
					with wave.open(wav_buffer, 'wb') as wf:
						wf.setnchannels(self.channels)
						wf.setsampwidth(2)
						wf.setframerate(self.sample_rate)
						wf.writeframes(audio_data.tobytes())
					
					# Send for transcription
					response = requests.post(
						self.whisper_url,
						headers=self.headers,
						data=wav_buffer.getvalue(),
						timeout=30
					)
					
					if response.status_code == 200:
						result = response.json()
						if "text" in result:
							self.current_transcription += result["text"] + " "
					
					# Clear accumulated audio but keep last chunk for continuity
					accumulated_audio = [accumulated_audio[-1]]
					
				except Exception as e:
					self.logger.error(f"Streaming transcription error: {str(e)}")

	def __save_recording(self, filename):
		"""Save the recording with proper WAV formatting and verification."""
		if not self.recording:
			self.logger.error("No audio data recorded")
			return None

		try:
			audio_data = np.concatenate(self.recording, axis=0)
			audio_data = (audio_data * np.iinfo(np.int16).max).astype(np.int16)
			
			with wave.open(filename, 'wb') as wf:
				wf.setnchannels(self.channels)
				wf.setsampwidth(2)
				wf.setframerate(self.sample_rate)
				wf.writeframes(audio_data.tobytes())
			
			if os.path.exists(filename) and os.path.getsize(filename) > 0:
				self.logger.info(f"Recording saved successfully to {filename}")
				return filename
			else:
				raise Exception("Failed to save recording: File is empty or not created")
				
		except Exception as e:
			self.logger.error(f"Error saving recording: {str(e)}")
			raise

	def process_audio_stream(self, session_id):
		"""Continuously process audio chunks from the buffer"""
		buffer = self.audio_buffer[session_id]
		accumulated_audio = b''

		print("streaming")
		
		while True:
			try:
				chunk = buffer.get(timeout=1.0)
				if chunk == b'STOP':
					# Process final chunk
					if accumulated_audio:
						text = self.transcribe_chunk(accumulated_audio)
						socketio.emit('transcription', {'text': text}, room=session_id)
					break
				
				accumulated_audio += chunk
				
				# Process accumulated audio every 3 seconds worth of data
				if len(accumulated_audio) >= self.sample_rate * 3 * 2:  # 3 seconds of 16-bit audio
					text = self.transcribe_chunk(accumulated_audio)
					socketio.emit('transcription', {'text': text}, room=session_id)
					accumulated_audio = b''
					
			except queue.Empty:
				# Process any remaining audio if we haven't received new data
				if accumulated_audio:
					text = self.transcribe_chunk(accumulated_audio)
					socketio.emit('transcription', {'text': text}, room=session_id)
					accumulated_audio = b''

	def transcribe_chunk(self, audio_data):
		"""Transcribe a chunk of audio data"""
		try:
			# Convert audio data to WAV format
			wav_buffer = io.BytesIO()
			with wave.open(wav_buffer, 'wb') as wf:
				wf.setnchannels(self.channels)
				wf.setsampwidth(2)
				wf.setframerate(self.sample_rate)
				wf.writeframes(audio_data)
			
			wav_data = wav_buffer.getvalue()
			
			# Send to Whisper API
			response = requests.post(
				self.whisper_url,
				headers=self.headers,
				data=wav_data,
				timeout=30
			)
			response.raise_for_status()
			result = response.json()
			return result.get("text", "")
			
		except Exception as e:
			self.logger.error(f"Transcription error: {e}")
			return ""
	

def main():
	"""Main function with environment validation and error handling."""
	try:
		load_dotenv(".env")
		whisper_url = os.environ["WHISPER_URL"]
		hf_token = os.environ["HF_TOKEN"]
		
		if not whisper_url or not hf_token:
			raise ValueError("Missing required environment variables: WHISPER_URL or HF_TOKEN")
			
		stt_service = SpeechToText(whisper_url, hf_token)
		
		output_dir = Path("./data/audio")
		output_dir.mkdir(parents=True, exist_ok=True)
		output_file = str(output_dir / "output.wav")
		
		filename = stt_service.start_recording(output_file)
		if filename:
			print("\nFinal transcription:", stt_service.current_transcription)
	
	except Exception as e:
		logging.error(f"Application error: {str(e)}")
		raise

if __name__ == "__main__":
	main()
