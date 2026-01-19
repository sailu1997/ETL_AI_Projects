"""
Document Retriever
Handles context retrieval for RAG queries.
"""

from typing import List, Dict
from .vector_store import VectorStoreManager


class DocumentRetriever:
    """Retrieve relevant context for user queries."""
    
    def __init__(self, vector_store_path: str):
        """
        Initialize retriever.
        
        Args:
            vector_store_path: Path to FAISS index
        """
        self.vector_store_manager = VectorStoreManager(vector_store_path)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of dictionaries containing document content and metadata
        """
        results = self.vector_store_manager.search(query, top_k=top_k)
        
        retrieved_docs = []
        for doc, score in results:
            retrieved_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            })
        
        return retrieved_docs
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents into context string for LLM.
        
        Args:
            retrieved_docs: List of retrieved document dictionaries
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant context found."
        
        context_parts = []
        for idx, doc in enumerate(retrieved_docs, 1):
            source = doc['metadata'].get('file_name', 'Unknown')
            content = doc['content']
            score = doc['score']
            
            context_parts.append(
                f"[Source {idx}: {source} (relevance: {score:.2f})]\n{content}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def get_stats(self) -> dict:
        """Get retriever statistics."""
        return self.vector_store_manager.get_stats()
