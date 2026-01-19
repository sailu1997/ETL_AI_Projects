"""
Main LangGraph Backend Server

This runs the LangGraph backend with FastAPI integration for file processing.
"""

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_integration import create_langgraph_endpoints

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="LangGraph File Processing Backend",
    description="Backend for file upload, cleaning, and processing using LangGraph workflows",
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

# Add LangGraph endpoints
create_langgraph_endpoints(app)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "langgraph-file-processing",
        "version": "1.0.0"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "LangGraph File Processing Backend",
        "version": "1.0.0",
        "endpoints": {
            "insights": "/insights/",
            "clean_data": "/clean_data/",
            "files": "/files/list",
            "session_create": "/session/create",
            "session_validate": "/session/validate",
            "session_info": "/session/info",
            "session_deactivate": "/session/deactivate",
            "chat": "/chat_stream",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    # Create upload directory if it doesn't exist
    os.makedirs("uploaded_files", exist_ok=True)
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,  # Use port 8000 for consistency
        reload=True
    ) 