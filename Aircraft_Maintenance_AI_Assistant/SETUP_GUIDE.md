# Setup Guide

Complete setup instructions for the Aircraft Maintenance AI Assistant.

## Prerequisites

- Python 3.9+
- Azure OpenAI API access
- Poppler (for PDF processing)
- Tesseract OCR

### Install System Dependencies

**macOS:**
```bash
brew install poppler tesseract
```

**Ubuntu:**
```bash
sudo apt-get install poppler-utils tesseract-ocr
```

**Windows:**
- Download Poppler from https://poppler.freedesktop.org/
- Download Tesseract from https://github.com/UB-Mannheim/tesseract/wiki

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd Aircraft_Maintenance_AI_Assistant
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp env_template.txt .env
```

Edit `.env` with your credentials:

```bash
AZURE_OPENAI_ENDPOINT=<your-endpoint>
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_VISION_DEPLOYMENT=<deployment-name>
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=<deployment-name>
AZURE_OPENAI_LLM_DEPLOYMENT=<deployment-name>
```

## Running the System

### Data Pipeline

Process maintenance documents:

```bash
cd 1_data_pipeline

# Split PDFs
python scripts/1_split_pdf.py --input data/raw/MM_Index.pdf --output data/split_pdfs/

# Extract sections
python scripts/2_extract_sections.py --input data/split_pdfs/ --output data/extracted/MM.csv --doc-type MM

# Post-process
python scripts/3_postprocess.py --input data/extracted/MM.csv --output data/processed/MM_formatted.csv --doc-type MM

# Generate embeddings
python scripts/4_generate_embeddings.py --input data/processed/MM_formatted.csv --output data/vector_stores/mm_faiss_index
```

### Backend API

Start the FastAPI service:

```bash
cd 2_backend
python api/app.py
```

Verify it's running:
```bash
curl http://localhost:8000/health
```

### Frontend

Launch the Streamlit app:

```bash
cd 3_frontend
streamlit run Main.py
```

Access at `http://localhost:8501`

## Testing

Test the backend API:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is MEL 21-33?", "top_k": 5}'
```

## Troubleshooting

### PDF Conversion Fails

Ensure Poppler is installed and in your PATH.

### Tesseract Not Found

Install Tesseract OCR and set `TESSDATA_PREFIX` if needed.

### Azure OpenAI Rate Limits

Reduce parallel workers in extraction script or add delays between requests.

### FAISS Memory Issues

Use smaller chunk sizes when generating embeddings or process documents in batches.

## Deployment

### Docker

Each module includes a Dockerfile for containerized deployment.

```bash
# Backend
cd 2_backend
docker build -t maintenance-ai-backend .
docker run -p 8000:8000 --env-file ../.env maintenance-ai-backend

# Frontend
cd 3_frontend
docker build -t maintenance-ai-frontend .
docker run -p 8501:8501 --env-file ../.env maintenance-ai-frontend
```

### Production

For production deployment:
- Use Azure Container Apps or AWS ECS for backend
- Deploy frontend on Streamlit Cloud or Azure App Service
- Store vector indices in Azure Blob Storage or S3
- Manage secrets with Azure Key Vault or AWS Secrets Manager
- Add monitoring and logging
- Implement authentication and authorization

## Next Steps

1. Process additional document types (MEL, CDL, FTD)
2. Combine multiple vector stores
3. Add user authentication
4. Implement query caching
5. Set up CI/CD pipeline
6. Add monitoring and analytics
