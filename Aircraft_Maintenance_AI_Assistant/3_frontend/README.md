# Frontend - Streamlit Application

Multi-page Streamlit web application providing an interface for aircraft maintenance engineers to interact with the AI assistant.

## Features

- **Main Dashboard**: Authentication and app navigation
- **Curie**: HR/policy assistant with custom knowledge base
- **Task Card Validator**: vLLM validation of maintenance task cards

## Setup

Install dependencies:

```bash
pip install streamlit st-pages loguru pandas langchain langchain-openai faiss-cpu openai
```

Configure environment:

```bash
BACKEND_API_URL=http://localhost:8000
AZURE_OPENAI_ENDPOINT=<your-endpoint>
AZURE_OPENAI_API_KEY=<your-key>
```

## Running the App

```bash
streamlit run Main.py
```

The app will open at `http://localhost:8501`.

## Usage

### Authentication

Enter your ID and department on the main page to access the tools.

### Task Card Validator

The Task Card Validator automates validation of aircraft maintenance task cards:

1. Upload a ZIP file containing taskcard PDFs
2. The system processes each PDF:
   - Converts pages to images
   - Uses OCR to detect relevant sections
   - Calls Azure OpenAI Vision to validate:
     - Signature present (YES/NO/NULL)
     - Stamp present (YES/NO/NULL)
     - Date completed (DD-MM-YYYY format)
3. Results are displayed in a table with PASS/FAIL/NULL status
4. Download results as CSV

**Background Processing**: The `background.py` script handles PDF processing asynchronously, allowing users to upload multiple batches without waiting.

## Architecture

```
User Interface (Streamlit)
       ↓
Backend API Calls
       ↓
RAG Pipeline
       ↓
Response Display
```

## File Structure

- `Main.py` - Entry point and authentication
- `common.py` - Shared utilities and configurations
- `config.py` - API configuration
- `background.py` - Background task processor for Task Card Validator
- `hr_ex.py` - HR-related utilities
- `1_Taskcard_Validator.py` - Task card validation interface
- `2_Curie.py` - HR policy assistant

## Deployment

### Docker

```bash
docker build -t maintenance-ai-frontend .
docker run -p 8501:8501 --env-file .env maintenance-ai-frontend
```

### Production Considerations

- Integrate with corporate SSO for authentication
- Add user activity logging
- Implement rate limiting
- Add caching for common queries
- Set up monitoring and performance metrics


