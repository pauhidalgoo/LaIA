import json
from openai import OpenAI
import logging
import requests
import re

# Set up logging
logger = logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SelectBestSources:
	def __init__(self, base_url: str, api_key: str, max_source_length: int = 500) -> None:
		self.client = OpenAI(
			base_url=base_url + "/v1/",
			api_key=api_key
		)
		self.max_source_length = max_source_length
		self.gathered_info = None

	def set_gathered_info(self, gathered_info_json_path: str) -> None:
		self.gathered_info = json.load(open(gathered_info_json_path, 'r'))

	def url_in_gathered_info(self, url: str) -> bool:
		def aux(source: dict):
			if source["url"] == url:
				return True
			elif source["sub_content"]:
				return any(aux(sub_source) for sub_source in source["sub_content"])
			else:
				return False
		
		sources = self.gathered_info["gathered_info"]
		return any(aux(source) for source in sources)

	def valid_url(self, url: str) -> bool:
		# Check if the URL is inside the json file
		if not self.url_in_gathered_info(url):
			return False
		
		# Check if the URL is valid
		try:
			r = requests.head(url, allow_redirects=True)
			if r.status_code == 200:
				return True
		except requests.RequestException as e:
			logger.warning(f"Failed to reach {url}: {str(e)}")

		return False
			
	def select_best_sources(self, query: str, gathered_info_json_path: str) -> list[str]:
		context = f"Query: {query}\n\nSources and gathered information:\n"

		self.set_gathered_info(gathered_info_json_path)

		for info in self.gathered_info:
			context += f"\nSource: {info['url']}\n"
			context += f"Content: {info['main_content'][:self.max_source_length]}...\n"
		
		messages = [
			{"role": "system", "content": "You are a helpful assistant that synthesizes information from multiple sources to provide accurate and comprehensive answers. You will be given a query and a list of sources with their main content. Your goal is to provide a list of the sources that contain the information needed to answer the query."},
			{"role": "user", "content": f"Based on the following information, provide a list of the sources that satisfy the query. Provide all the URLs needed (between the text).{context}"}
		]
		data_to_save = {
			"query": query,
			"gathered_info": self.gathered_info,
			"messages": messages
		}

		# Define the file path
		file_path = "./data/context_data.json"

		# Save the dictionary as a JSON file
		with open(file_path, 'w') as json_file:
			json.dump(data_to_save, json_file, indent=4)

		print(f"Data saved to {file_path}")

		try:
			response = self.client.chat.completions.create(
				model="tgi",
				messages=messages,
				temperature=0.1,
				max_tokens=1000
			)
			
			urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', response.choices[0].message['content'])
			valid_urls = [url for url in urls if self.valid_url(url, gathered_info_json_path)]
			
			return valid_urls
			
		except Exception as e:
			logger.error(f"Error in synthesis: {str(e)}")
			return "Error synthesizing information from sources."