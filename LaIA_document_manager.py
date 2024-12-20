from dataclasses import dataclass
from typing import Optional, List, Dict, Union
from datetime import datetime
import uuid
import json
from langchain_community.vectorstores import FAISS

@dataclass
class Document:
    id: str
    title: str
    type: str  # 'pdf', 'image', 'web'
    content: str
    chunks: List[str]
    metadata: Dict
    timestamp: str
    source_url: Optional[str] = None

class DocumentManager:
    def __init__(self, embeddings, text_splitter):
        self.documents: Dict[str, Document] = {}
        self.vector_store = None
        self.embeddings = embeddings
        self.text_splitter = text_splitter
        
    def add_document(self, 
                    title: str, 
                    content: str, 
                    doc_type: str, 
                    metadata: Dict = None,
                    source_url: Optional[str] = None) -> Document:
        """Add a new document to the manager and update vector store"""
        doc_id = str(uuid.uuid4())
        chunks = self.text_splitter.split_text(content)
        
        document = Document(
            id=doc_id,
            title=title,
            type=doc_type,
            content=content,
            chunks=chunks,
            metadata=metadata or {},
            timestamp=datetime.now().isoformat(),
            source_url=source_url
        )
        
        self.documents[doc_id] = document

        all_chunks = []
        metadatas = []
        ids = []
        for i, chunk in enumerate(document.chunks):
            all_chunks.append(chunk)
            metadatas.append({
                'doc_id': document.id,
                'doc_title': document.title,
                'doc_type': document.type,
                'chunk_index': i,
                'source_url': document.source_url
            })
            ids.append(f"{document.id}_{i}")  # Assign unique IDs to chunks

        if self.vector_store is not None:
            self.vector_store.add_texts(all_chunks, metadatas=metadatas, ids=ids)
        else:
            self.vector_store = FAISS.from_texts(all_chunks, self.embeddings, metadatas=metadatas, ids=ids)

        self.vector_store.save_local("faiss_index")
        return document, len(chunks)

    

    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a document and update vector store"""
        if doc_id in self.documents:
            # Remove document from documents
            del self.documents[doc_id]

            # Collect IDs of chunks to delete
            ids_to_delete = [
                vector_id for vector_id, doc in self.vector_store.docstore._dict.items()
                if doc.metadata.get('doc_id') == doc_id
            ]

            if ids_to_delete:
                self.vector_store.delete(ids_to_delete)

            return True
        return False

   
    
    def search(self, query: str, k: int = 3, relevance_threshold: float = 1.5, llm_client = None) -> List[Dict]:
        """Search for relevant chunks across all documents"""
        if not self.vector_store:
            return []
        
            
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        formatted_results = [
            {
                'content': doc.page_content,
                'doc_title': doc.metadata['doc_title'],
                'doc_type': doc.metadata['doc_type'],
                'source_url': doc.metadata.get('source_url'),
                'relevance_score': float(score)
            }
            for doc, score in results
        ]
        print(formatted_results)
        formatted_results = [d for d in formatted_results]
            
        return formatted_results
    
    def get_document_list(self) -> List[Dict]:
        """Get a list of all documents for the sidebar"""
        return [
            {
                'id': doc.id,
                'title': doc.title,
                'type': doc.type,
                'timestamp': doc.timestamp,
                'chunk_count': len(doc.chunks),
                'source_url': doc.source_url
            }
            for doc in self.documents.values()
        ]
    
    def get_context(self, 
                        query: str,
                        llm_client,
                        k: int = 2,
                        include_citations: bool = True) -> Dict:
        """Generate a response using RAG with citations"""
        relevant_chunks = self.search(query, k=k, llm_client= llm_client)
        
        if not relevant_chunks:
            return {
                'response': 'No relevant context found to answer the question.',
                'citations': []
            }
            
        # Prepare context with citations
        context = "\n\n".join([
            f"{chunk['content']}"
            for i, chunk in enumerate(relevant_chunks)
        ])
        
        return context

    def generate_response(self, 
                         query: str,
                         llm_client,
                         k: int = 3,
                         include_citations: bool = True) -> Dict:
        """Generate a response using RAG with citations"""
        relevant_chunks = self.search(query, k=k, llm_client= llm_client)
        
        if not relevant_chunks:
            return {
                'response': 'No relevant context found to answer the question.',
                'citations': []
            }
            
        # Prepare context with citations
        context = "\n\n".join([
            f"[{i+1}] {chunk['content']}"
            for i, chunk in enumerate(relevant_chunks)
        ])
        
        messages = [
            {"role": "system", "content": """You are a helpful assistant that answers questions based on the provided context. 
             Include citation numbers [1], [2], etc. when referencing specific information from the context."""},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query} If the context is unrelated, return NOT_FOUND."}
        ]

        
        
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini", # Old was tgi
            max_tokens=1000,
            messages=messages,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content

        if "NOT_FOUND" in answer or "no he trobat" in answer.lower():
            print(response.choices[0].message.content)
            return {
                'response': 'No relevant context found to answer the question.',
                'citations': []
            }
        
        return {
            'response': response.choices[0].message.content,
            'citations': [
                {
                    'number': i + 1,
                    'doc_title': chunk['doc_title'],
                    'doc_type': chunk['doc_type'],
                    'source_url': chunk['source_url'],
                    'content': chunk['content'][:200] + '...'  # Preview
                }
                for i, chunk in enumerate(relevant_chunks)
            ]
        }