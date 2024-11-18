import json
from openai import OpenAI
import logging
import requests
import re

# Set up logging
logger = logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SelectBestSources:
	def __init__(self, base_url: str, api_key: str, max_source_chars_length: int = 500, max_simultaneous_sources: int = 5, remove_parent_urls: bool = False) -> None:
		self.client = OpenAI(
			#base_url=base_url + "/v1/",
			api_key=api_key
		)
		self.max_source_chars_length = max_source_chars_length
		self.max_simultaneous_sources = max_simultaneous_sources
		self.remove_parent_urls = remove_parent_urls
		self.gathered_info = None
		self.query = None
		self.selected_sources = set()

	def __set_gathered_info(self, gathered_info_json_path: str) -> None:
		json_content = json.load(open(gathered_info_json_path, 'r'))
		self.gathered_info = json_content["gathered_info"]
		self.query = json_content["query"]

	def __valid_url(self, url: str) -> bool:
		# Check if the URL is valid
		try:
			r = requests.head(url, allow_redirects=True)
			if r.status_code == 200:
				return True
		except requests.RequestException as e:
			logger.warning(f"Failed to reach {url}: {str(e)}")

		return False
	
	def __get_sources_list_from_gathered_info(self) -> list[tuple]:
		def aux(source: dict):
			if source["sub_content"]:
				current = [(source["url"], source["main_content"])]
				subs = [aux(sub_source) for sub_source in source["sub_content"]]

				for sub in subs:
					current += sub

				return current
			
			else:
				return [(source["url"], source["main_content"])]
		
		results = []
		for source in self.gathered_info:
			results += aux(source)

		return results
			
	def __select_best_sources(self, sources: list[tuple]) -> set[tuple]:
		assert self.gathered_info is not None, "Gathered information is not set. Please set it using the set_gathered_info method."

		context = f"Query: {self.query}\n\nSources and gathered information:\n"

		for info in sources:
			context += f"\nSource: {info[0]}\n"
			context += f"Content: {info[1][:self.max_source_chars_length]}...\n"
		
		messages = [
			{"role": "system", "content": "You are a helpful assistant that synthesizes information from multiple sources to provide accurate and comprehensive answers. You will be given a query and a list of sources with their main content. Your goal is to provide a list of the sources that contain the information needed to answer the query. Always return all the URLs needed."},
			{"role": "user", "content": f"Based on the following information, provide a list of the sources that most satisfy the query. Provide all the URLs needed (between the text).{context}"}
		]

		try:
			response = self.client.chat.completions.create(
				model="gpt-4o-mini", # Old was tgi
				messages=messages,
				temperature=0.1,
				max_tokens=1000,
				frequency_penalty=0.2
			)

			answer = response.choices[0].message.content

			selected_urls = re.findall(r"https?://[^\s,)}\]]+", answer)

			selected_urls = [re.sub(r"[,)}\]]+$", "", url) for url in selected_urls]
			
			all_urls = [s[0] for s in sources]
			all_contents = [s[1] for s in sources]
			
			valid_urls = []
			for url in selected_urls:
				if url in all_urls and self.__valid_url(url):
					valid_urls.append(url)

			valid_urls_and_contents = set([(url, all_contents[all_urls.index(url)]) for url in valid_urls])

			if self.remove_parent_urls:
				valid_urls_and_contents = self.__remove_parent_urls_from_set(valid_urls_and_contents)

			return valid_urls_and_contents
			
		except Exception as e:
			logger.error(f"Error in synthesis: {str(e)}")
			return []
		
	def __remove_parent_urls_from_set(self, urls_and_contents: set[tuple]) -> set[tuple]:
		if not urls_and_contents:
			return urls_and_contents

		sorted_urls = list(sorted(urls_and_contents, key=lambda x: len(x[0]), reverse=False))

		removed_indices = []
		for i, (url, _) in enumerate(sorted_urls):
			for (parent_url, _) in sorted_urls[i + 1:]:
				if parent_url.startswith(url):
					removed_indices.append(i)

		return set([sorted_urls[i] for i in range(len(sorted_urls)) if i not in removed_indices])
		
	def get_current_sources(self) -> list[tuple]:
		return list(self.selected_sources)
	
	def reset_current_sources(self) -> None:
		self.selected_sources = set()
	
	def append_sources(self, gathered_info_json_path: str) -> None:
		self.__set_gathered_info(gathered_info_json_path)

		sources_list = self.__get_sources_list_from_gathered_info()

		for sources_sublist in [sources_list[i:i + self.max_simultaneous_sources] for i in range(0, len(sources_list), self.max_simultaneous_sources)]:
			best_sources = self.__select_best_sources(sources=sources_sublist)

			self.selected_sources.update(best_sources)

			if self.remove_parent_urls:
				self.selected_sources = self.__remove_parent_urls_from_set(self.selected_sources)

	def get_final_sources(self, original_query: str) -> list[tuple]:
		self.query = original_query
		return list(self.__select_best_sources(sources=list(self.selected_sources)))
	

# Example usage

from dotenv import load_dotenv
import os

def main():
	load_dotenv(".env")
	
	# Initialize the agent
	base_url = os.environ["BASE_URL"]
	api_key = os.environ["HF_TOKEN"]
	
	select_best_sources = SelectBestSources(base_url=base_url, api_key=api_key, max_source_chars_length=500, max_simultaneous_sources=5, remove_parent_urls=True)

	gathered_info_json_path = "./data/context_data_COM PUC FER-ME PROFE.json"

	select_best_sources.append_sources(gathered_info_json_path=gathered_info_json_path)

	for source in select_best_sources.get_current_sources():
		print(f"{source[0]}")
	

if __name__ == "__main__":
	main()
