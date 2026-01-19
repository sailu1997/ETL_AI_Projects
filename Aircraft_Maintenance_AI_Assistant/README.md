# Aircraft Maintenance AI Assistant

An end-to-end AI system for processing Boeing 737 maintenance documentation and enabling intelligent query-answering for aircraft engineers.

## Problem Statement

Aircraft maintenance engineers need quick access to critical information from thousands of pages of technical documentation (Maintenance Memos, MEL, CDL, FTD). Manual search is time-consuming and error-prone, potentially impacting aircraft safety and operational efficiency.

## Solution

This system processes maintenance PDFs using vLLM , generates vector embeddings for semantic and keyword based search, and provides a conversational interface for engineers to query documentation using natural language.

## Architecture

The system consists of three main components:

1. **Data Pipeline** - Extracts structured data from PDFs using Azure OpenAI api, processes and formats the data, then generates FAISS vector embeddings
2. **Backend API** - FastAPI service that handles query embedding, vector search, and LLM-based response generation (RAG architecture)
3. **Frontend** - Streamlit web application with multiple specialized tools for different maintenance tasks

## Repository Structure

```
├── 1_data_pipeline/          # ETL: PDF processing & embeddings
│   ├── scripts/               # Production Python scripts
│   │   ├── 1_split_pdf.py
│   │   ├── 2_extract_sections.py
│   │   ├── 3_postprocess.py
│   │   └── 4_generate_embeddings.py
│   └── prompts/               # LLM extraction prompts
│
├── 2_backend/                 # RAG service
│   ├── retrieval/             # Vector search & context retrieval
│   ├── generation/            # LLM response generation
│   └── api/                   # FastAPI REST endpoints
│
├── 3_frontend/                # Streamlit application
│   ├── Main.py                # Entry point & authentication
│   ├── 1_Taskcard_Validator.py
│   ├── 2_Curie.py
│   └── background.py          # Async task processing
│
└── requirements.txt           # Python dependencies
```

## Quick Start

### Prerequisites

- Python 3.9+
- Azure OpenAI API access (GPT-4 Vision, GPT-4, text-embedding-ada-002)
- Poppler and Tesseract OCR installed

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env_template.txt .env
# Edit .env with your API credentials
```

### Running the Pipeline

```bash
# 1. Split multi-document PDFs
cd 1_data_pipeline
python scripts/1_split_pdf.py --input data/raw/MM_Index.pdf --output data/split_pdfs/

# 2. Extract structured data
python scripts/2_extract_sections.py --input data/split_pdfs/ --output data/extracted/MM.csv --doc-type MM

# 3. Post-process for embeddings
python scripts/3_postprocess.py --input data/extracted/MM.csv --output data/processed/MM_formatted.csv --doc-type MM

# 4. Generate vector embeddings
python scripts/4_generate_embeddings.py --input data/processed/MM_formatted.csv --output data/vector_stores/mm_faiss_index
```

### Starting the Backend

```bash
cd 2_backend
python api/app.py
```

### Launching the Frontend

```bash
cd 3_frontend
streamlit run Main.py
```

## Key Features

- **Vision-based Extraction**: Handles complex tables and diagrams in maintenance documents
- **Semantic Search**: FAISS-based vector retrieval for intelligent document search
- **RAG Architecture**: Context-aware responses using retrieved documentation
- **Task Card Validator**: Automated validation of maintenance task cards using Vision AI
- **Multi-tool Interface**: Specialized modules for different maintenance tasks

## Technology Stack

- **Data Processing**: pdf2image, pytesseract, PyPDF2
- **AI/ML**: Azure OpenAI (GPT-4 Vision, GPT-4, embeddings), LangChain, FAISS
- **Backend**: FastAPI, Pydantic
- **Frontend**: Streamlit
- **Data**: pandas, numpy

## Environment Variables

Required environment variables (see `env_template.txt`):

```bash
AZURE_OPENAI_ENDPOINT=<your-endpoint>
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_VISION_DEPLOYMENT=<deployment-name>
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=<deployment-name>
AZURE_OPENAI_LLM_DEPLOYMENT=<deployment-name>
```

## Documentation

- [Data Pipeline README](1_data_pipeline/README.md) - ETL process details
- [Backend README](2_backend/README.md) - API documentation
- [Frontend README](3_frontend/README.md) - Application guide
- [Setup Guide](SETUP_GUIDE.md) - Complete installation instructions

## Project Highlights

- Processes multiple document types (MM, MEL, CDL, FTD)
- Parallel processing with ThreadPoolExecutor for faster extraction
- Production-ready error handling and retry logic
- Modular architecture for easy extension
- Task Card Validator uses Vision AI to automatically check signatures, stamps, and completion dates

## License

This project was developed for aircraft maintenance operations.
