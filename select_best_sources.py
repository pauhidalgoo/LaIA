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
		self.selected_sources = []

	def __set_gathered_info(self, gathered_info_json_path: str) -> None:
		self.gathered_info = json.load(open(gathered_info_json_path, 'r'))

	def __valid_url(self, url: str) -> bool:
		# Check if the URL is valid
		try:
			r = requests.head(url, allow_redirects=True)
			if r.status_code == 200:
				return True
		except requests.RequestException as e:
			logger.warning(f"Failed to reach {url}: {str(e)}")

		return False
	
	def __get_sources_list_from_gathered_info(self) -> list[str]:
		def aux(source: dict):
			if source["sub_content"]:
				return [(source["url"], source["main_content"])] + [aux(sub_source) for sub_source in source["sub_content"]]
			else:
				return [(source["url"], source["main_content"])]
		
		sources = self.gathered_info["gathered_info"]
		
		return [aux(source) for source in sources]
			
	def __select_best_sources(self, query: str, sources: list[dict]) -> list[tuple]:
		assert self.gathered_info is not None, "Gathered information is not set. Please set it using the set_gathered_info method."

		context = f"Query: {query}\n\nSources and gathered information:\n"

		for info in sources:
			context += f"\nSource: {info[0]}\n"
			context += f"Content: {info[1][:self.max_source_length]}...\n"
		
		messages = [
			{"role": "system", "content": "You are a helpful assistant that synthesizes information from multiple sources to provide accurate and comprehensive answers. You will be given a query and a list of sources with their main content. Your goal is to provide a list of the sources that contain the information needed to answer the query."},
			{"role": "user", "content": f"Based on the following information, provide a list of the sources that satisfy the query. Provide all the URLs needed (between the text).{context}"}
		]

		try:
			response = self.client.chat.completions.create(
				model="tgi",
				messages=messages,
				temperature=0.1,
				max_tokens=1000
			)
			
			selected_urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', response.choices[0].message['content'])

			all_urls = [s[0] for s in sources]
			all_contents = [s[1] for s in sources]
			
			valid_urls = []
			for url in selected_urls:
				if url in all_urls and self.__valid_url(url):
					valid_urls.append(url)

			valid_urls_and_contents = [(url, all_contents[all_urls.index(url)]) for url in valid_urls]
			
			return valid_urls_and_contents
			
		except Exception as e:
			logger.error(f"Error in synthesis: {str(e)}")
			return "Error synthesizing information from sources."
		
	def get_current_sources(self) -> list[tuple]:
		return self.selected_sources
	
	def reset_current_sources(self) -> None:
		self.selected_sources = []
	
	def append_sources(self, query: str, gathered_info_json_path: str) -> None:
		self.__set_gathered_info(gathered_info_json_path)

		sources_list = self.__get_sources_list_from_gathered_info()

		best_sources = self.__select_best_sources(query=query, sources=sources_list)

		self.selected_sources += best_sources

	def get_final_sources(self, original_query: str) -> list[tuple]:
		return self.__select_best_sources(query=original_query, sources=self.selected_sources)
