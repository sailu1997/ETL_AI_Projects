"""
LLM Client for Response Generation
Handles Azure OpenAI API calls for generating responses.
"""

import os
from typing import List, Dict
from openai import AzureOpenAI
from dotenv import load_dotenv
from .prompts import get_system_prompt

load_dotenv()


class LLMClient:
    """Azure OpenAI LLM client for generating responses."""
    
    def __init__(self, model: str = None):
        """
        Initialize LLM client.
        
        Args:
            model: Azure OpenAI deployment name (defaults to env var)
        """
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        )
        self.model = model or os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-4-32k")
    
    def generate(
        self,
        query: str,
        context: str,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        task_type: str = "general"
    ) -> Dict:
        """
        Generate a response based on query and context.
        
        Args:
            query: User query
            context: Retrieved context string
            temperature: Generation temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            task_type: Type of task (general, technical, troubleshooting)
            
        Returns:
            Dictionary with response and metadata
        """
        system_prompt = get_system_prompt(task_type)
        user_prompt = self._build_user_prompt(query, context)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return {
            "response": response.choices[0].message.content,
            "model": self.model,
            "tokens_used": response.usage.total_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens
        }
    
    def _build_user_prompt(self, query: str, context: str) -> str:
        """Build user prompt with query and context."""
        return f"""Based on the following context from aircraft maintenance documentation, answer the user's question.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""
    
    def generate_with_retrieved_docs(
        self,
        query: str,
        retrieved_docs: List[Dict],
        temperature: float = 0.0,
        task_type: str = "general"
    ) -> Dict:
        """
        Generate response from retrieved documents.
        
        Args:
            query: User query
            retrieved_docs: List of retrieved document dictionaries
            temperature: Generation temperature
            task_type: Type of task
            
        Returns:
            Dictionary with response, sources, and metadata
        """
        from ..retrieval.retriever import DocumentRetriever
        
        # Format context
        context = DocumentRetriever.format_context(None, retrieved_docs)
        
        # Generate response
        result = self.generate(query, context, temperature, task_type=task_type)
        
        # Add source information
        result["sources"] = [
            {
                "file_name": doc["metadata"].get("file_name", "Unknown"),
                "score": doc["score"]
            }
            for doc in retrieved_docs
        ]
        
        return result
