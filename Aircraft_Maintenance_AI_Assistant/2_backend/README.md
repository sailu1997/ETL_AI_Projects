# Backend - RAG Service

The backend implements a Retrieval-Augmented Generation (RAG) system for querying aircraft maintenance documentation.

## Architecture

```
User Query → Query Embedding → Vector Search → Context Retrieval → LLM Generation → Response
```

## Components

### Retrieval (`retrieval/`)

- **vector_store.py**: Manages FAISS vector store loading and searching
- **retriever.py**: Handles document retrieval and context formatting

### Generation (`generation/`)

- **llm_client.py**: Azure OpenAI LLM client for response generation
- **prompts.py**: System prompts for different task types (general, technical, troubleshooting, MEL/CDL)

### API (`api/`)

- **app.py**: FastAPI REST API with query endpoint

## Setup

Install dependencies:

```bash
pip install fastapi uvicorn pydantic langchain langchain-openai faiss-cpu openai python-dotenv
```

Configure environment:

```bash
AZURE_OPENAI_ENDPOINT=<your-endpoint>
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=<deployment-name>
AZURE_OPENAI_LLM_DEPLOYMENT=<deployment-name>
VECTOR_STORE_PATH=../1_data_pipeline/data/vector_stores/mm_faiss_index
API_PORT=8000
```

## Running the API

```bash
python api/app.py
```

Or with uvicorn:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### POST /query

Main query endpoint for RAG-based question answering.

**Request:**
```json
{
  "query": "What is the procedure for MEL 21-33?",
  "top_k": 5,
  "temperature": 0.0,
  "task_type": "general"
}
```

**Response:**
```json
{
  "query": "What is the procedure for MEL 21-33?",
  "response": "According to MEL 21-33...",
  "sources": [
    {"file_name": "MM_17.pdf", "score": 0.92}
  ],
  "tokens_used": 1234,
  "model": "gpt-4-32k"
}
```

### GET /health

Health check endpoint.

### GET /stats

Returns vector store statistics.

## Usage Example

```python
from retrieval.retriever import DocumentRetriever
from generation.llm_client import LLMClient

# Initialize
retriever = DocumentRetriever("data/vector_stores/mm_faiss_index")
llm = LLMClient()

# Process query
query = "What is the procedure for MEL 21-33?"
docs = retriever.retrieve(query, top_k=5)
response = llm.generate_with_retrieved_docs(query, docs)

print(response["response"])
print(response["sources"])
```

## Performance

- Embedding generation: ~100ms per query
- Vector search: ~50ms for top-5 retrieval
- LLM generation: 2-5 seconds
- Total latency: ~3-6 seconds per query

## Integration

The frontend (module 3) calls this API to process user queries. The API handles:
1. Query embedding
2. Vector similarity search
3. Context retrieval
4. LLM-based response generation
5. Source citation

## Task Types

The system supports different prompt styles based on task type:
- `general`: General maintenance queries
- `technical`: Detailed technical information
- `troubleshooting`: Diagnostic procedures
- `mel_cdl`: MEL/CDL dispatch decisions
