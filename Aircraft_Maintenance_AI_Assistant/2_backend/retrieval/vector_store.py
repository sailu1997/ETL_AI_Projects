"""
FAISS Vector Store Management
Loads and manages FAISS indices for document retrieval.
"""

import os
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


class VectorStoreManager:
    """Manage FAISS vector store for document retrieval."""
    
    def __init__(self, vector_store_path: str):
        """
        Initialize vector store.
        
        Args:
            vector_store_path: Path to saved FAISS index directory
        """
        self.vector_store_path = vector_store_path
        self.embeddings = self._init_embeddings()
        self.vector_store = self._load_vector_store()
    
    def _init_embeddings(self) -> AzureOpenAIEmbeddings:
        """Initialize Azure OpenAI embeddings model."""
        return AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "text-embedding-ada-002"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
    
    def _load_vector_store(self) -> FAISS:
        """Load FAISS vector store from disk."""
        if not os.path.exists(self.vector_store_path):
            raise FileNotFoundError(f"Vector store not found at {self.vector_store_path}")
        
        print(f"Loading vector store from {self.vector_store_path}")
        return FAISS.load_local(
            self.vector_store_path,
            self.embeddings,
            allow_dangerous_deserialization=True  # Required for loading pickled data
        )
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[any, float]]:
        """
        Search for similar documents.
        
        Args:
            query: User query string
            top_k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        return results
    
    def get_stats(self) -> dict:
        """Get vector store statistics."""
        return {
            "path": self.vector_store_path,
            "index_size": self.vector_store.index.ntotal,
            "embedding_dimension": self.vector_store.index.d
        }
