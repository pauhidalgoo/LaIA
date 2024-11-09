import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from googlesearch import search
import concurrent.futures
import logging
import json
import re
from select_best_sources import SelectBestSources
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSearchAgent:
    def __init__(self, base_url: str, api_key: str, max_depth: int = 3, max_links_per_page: int = 5):
        """
        Initialize the web search agent with configuration parameters.
        
        Args:
            base_url: Base URL for the LLM API
            api_key: API key for authentication
            max_depth: Maximum depth for recursive link exploration
            max_links_per_page: Maximum number of links to explore per page
        """
        self.client = OpenAI(
            base_url=base_url + "/v1/",
            api_key=api_key
        )
        self.max_depth = max_depth
        self.max_links_per_page = max_links_per_page
        self.visited_urls = set()
        
    def search_and_analyze(self, query: str) -> str:
        """
        Main method to handle the search and analysis process.
        
        Args:
            query: User's search query
            
        Returns:
            str: Synthesized response based on gathered information
        """
        try:
            # Initial Google search
            # print("Searching " + query + " ...")
            search_results = list(search(query + " site:gencat.cat/ca", num_results=2))
            
            if not search_results:
                print("No search results found for the query in Gencat, searching in the whole web.")
                search_results = list(search(query, num_results=2))
                if not search_results:
                    return False
            
            # Collect information from multiple sources
            gathered_info = []
            for url in search_results:
                info = self._explore_url(url, depth=0)
                if info:
                    gathered_info.append(info)
            
            # Synthesize final response using LLM
            return self._synthesize_information(query, gathered_info)
            
        except Exception as e:
            logger.error(f"Error in search_and_analyze: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"

    def _explore_url(self, url: str, depth: int) -> Optional[Dict]:
        """
        Recursively explore a URL and its linked content.
        
        Args:
            url: URL to explore
            depth: Current exploration depth
            
        Returns:
            Optional[Dict]: Dictionary containing extracted information and metadata
        """
        print("Searching " + str(url) + "...")
        if depth >= self.max_depth or url in self.visited_urls or url.endswith(".pdf"):
            return None
            
        try:
            # Mark URL as visited
            self.visited_urls.add(url)
            
            # Fetch and parse content
            response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content
            main_content = self._extract_main_content(soup)
            
            # Extract relevant links for further exploration
            links = self._extract_relevant_links(soup, url)
            
            # Recursive exploration of linked content
            sub_content = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_to_url = {
                    executor.submit(self._explore_url, link, depth + 1): link 
                    for link in links[:self.max_links_per_page]
                }
                for future in concurrent.futures.as_completed(future_to_url):
                    result = future.result()
                    if result:
                        sub_content.append(result)
                executor.shutdown(wait=True)
            
            return {
                'url': url,
                'main_content': main_content,
                'sub_content': sub_content,
                'depth': depth
            }
            
        except Exception as e:
            logger.error(f"Error exploring URL {url}: {str(e)}")
            return None

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extract the main content from a webpage.
        
        Args:
            soup: BeautifulSoup object of the webpage
            
        Returns:
            str: Extracted main content
        """
        # Remove script and style elements
        for script in soup(['script', 'hidden-xs',  'style', 'nav', 'footer', 'header']):
            script.decompose()
            
        # Extract text from common content containers
        content_containers = soup.find_all(['article', 'div'], class_=['content', 'main', 'tramit-steps' , 'container', 'article', 'blocs'])
        
        if content_containers:
            return ' '.join(container.get_text(strip=True) for container in content_containers)
        
        # Fallback to body content if no specific containers found
        return soup.body.get_text(strip=True) if soup.body else ''

    def _extract_relevant_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Extract relevant links from a webpage.
        
        Args:
            soup: BeautifulSoup object of the webpage
            base_url: Base URL for resolving relative links
            
        Returns:
            List[str]: List of relevant URLs
        """
        links = []
        base_domain = urlparse(base_url).netloc
        
        for link in soup.find_all('a', href=True):
            url = urljoin(base_url, link['href'])
            parsed_url = urlparse(url)
            
            # Filter links based on relevance
            if (
                parsed_url.netloc == base_domain and  # Same domain
                parsed_url.scheme in ('http', 'https') and  # Valid scheme
                '#' not in url and  # Not an anchor
                'javascript:' not in url and  # Not JavaScript
                url not in self.visited_urls and # Not already visited
                ".pdf" not in url and
                not url.endswith(".pdf")

            ):
                links.append(url)
                
        return links

    def _synthesize_information(self, query: str, gathered_info: List[Dict]) -> str:
        """
        Use LLM to synthesize gathered information into a coherent response.
        
        Args:
            query: Original user query
            gathered_info: List of dictionaries containing gathered information
            
        Returns:
            str: Synthesized response
        """
        
        data_to_save = {
            "query": query,
            "gathered_info": gathered_info,
        }

        # Define the file path
        filename = re.sub(r'[\\/*?:"<>|]', '_', query.replace(' ', ''))
        file_path = f"./data/context_data_{filename}.json"

        # Save the dictionary as a JSON file
        with open(file_path, 'w') as json_file:
            json.dump(data_to_save, json_file, indent=4)
        return file_path


    def generate_rag_response(self, question: str, context: str) -> str:
        """
        Generates a response based on provided context using Retrieval Augmented Generation (RAG).

        Args:
            question: The user's question.
            context: Relevant context extracted from documents.

        Returns:
            str: The generated response.
        """
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
            ]

            response = self.client.chat.completions.create(
                model="tgi",  # Or your preferred model
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error in RAG response generation: {str(e)}")
            return "Error generating response based on provided context."
        
        
    def process_prompt(self, query: str) -> str:
        """
        Use LLM to synthesize gathered information into a coherent response.
        
        Args:
            query: Original user query
            gathered_info: List of dictionaries containing gathered information
            
        Returns:
            str: Synthesized response
        """
        # Prepare context for the LLM
        context = f"\nThe question is: {query}\n"
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that is very good searching the web. You don't explain your outputs, you only answer directly. You only speak Catalan. Examples: User: Soc un jove estudiant universitari que estic buscant quines són les millors beques per mi. \n Response: beques universitat. \n \n User: Vull anar a caçar amb el meu pare, què necessito \n Response: permisos necessaris caça."},
            {"role": "user", "content": f"Based on the following question, return what would you search in the Catalan administration website to find the most rellevant results. Only search for relevant administrative questions that may be useful for the user.\n {context}. Answer in Catalan and very concise, no explanation, only the query you would search. Very important: in Catalan. Return 3 different queries separated by an intro."}
        ]

        try:
            response = self.client.chat.completions.create(
                model="tgi",
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in synthesis: {str(e)}")
            return "Error synthesizing information from sources."



# Usage example
def main():
    load_dotenv(".env")
    
    base_url = os.environ["BASE_URL"]
    api_key = os.environ["HF_TOKEN"]
    # Initialize the agent
    agent = WebSearchAgent(
        base_url=base_url,
        api_key=api_key,
        max_depth=2,
        max_links_per_page=3
    )
    
    # Example query
    old_query = input("Enter your search query: ")
    
    query = agent.process_prompt(old_query)
    query = query.split("\n")
    responses = []

    select_best_sources = SelectBestSources(base_url=base_url, api_key=api_key, max_source_chars_length=500, max_simultaneous_sources=5, remove_parent_urls=True)
    for q in query[:3]:
    # Get response
        response = agent.search_and_analyze(q)
        select_best_sources.append_sources(response)
    responses = select_best_sources.get_final_sources(old_query)
    print(responses)
    

if __name__ == "__main__":
    main()