"""
FastAPI Backend for Aircraft Maintenance AI Assistant
Provides RAG (Retrieval-Augmented Generation) query endpoint.
"""

import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from retrieval.retriever import DocumentRetriever
from generation.llm_client import LLMClient

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Aircraft Maintenance AI Assistant API",
    description="RAG-based query API for aircraft maintenance documentation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize retriever and LLM client
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "../1_data_pipeline/data/vector_stores/mm_faiss_index")
retriever = DocumentRetriever(VECTOR_STORE_PATH)
llm_client = LLMClient()


# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    temperature: Optional[float] = 0.0
    task_type: Optional[str] = "general"


class QueryResponse(BaseModel):
    query: str
    response: str
    sources: list
    tokens_used: int
    model: str


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Aircraft Maintenance AI Assistant API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        stats = retriever.get_stats()
        return {
            "status": "healthy",
            "vector_store": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Main query endpoint for RAG-based question answering.
    
    Args:
        request: QueryRequest with query text and parameters
        
    Returns:
        QueryResponse with generated answer and sources
    """
    try:
        # Retrieve relevant documents
        retrieved_docs = retriever.retrieve(request.query, top_k=request.top_k)
        
        if not retrieved_docs:
            return QueryResponse(
                query=request.query,
                response="I couldn't find any relevant information in the documentation to answer your question.",
                sources=[],
                tokens_used=0,
                model=llm_client.model
            )
        
        # Generate response
        result = llm_client.generate_with_retrieved_docs(
            query=request.query,
            retrieved_docs=retrieved_docs,
            temperature=request.temperature,
            task_type=request.task_type
        )
        
        return QueryResponse(
            query=request.query,
            response=result["response"],
            sources=result["sources"],
            tokens_used=result["tokens_used"],
            model=result["model"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/stats")
async def get_statistics():
    """Get vector store statistics."""
    try:
        stats = retriever.get_stats()
        return {
            "vector_store": stats,
            "llm_model": llm_client.model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    print(f"Starting Aircraft Maintenance AI Assistant API on {host}:{port}")
    print(f"Vector Store: {VECTOR_STORE_PATH}")
    print(f"LLM Model: {llm_client.model}")
    
    uvicorn.run(app, host=host, port=port, reload=True)
