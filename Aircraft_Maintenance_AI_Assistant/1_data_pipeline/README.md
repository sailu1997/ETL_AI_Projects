# Data Pipeline

This module handles the ETL process for aircraft maintenance documentation.

## Overview

The pipeline extracts structured data from maintenance PDFs using Vision AI, processes and formats the data, then generates vector embeddings for semantic search.

## Workflow

```
Raw PDFs → Split → Vision Extraction → Post-processing → Embeddings → FAISS Index
```

## Scripts

### 1. Split PDF (`1_split_pdf.py`)

Splits multi-document PDFs into individual files by detecting "Page 1 of N" footers using OCR.

```bash
python scripts/1_split_pdf.py \
    --input data/raw/MM_Index.pdf \
    --output data/split_pdfs/ \
    --prefix MM
```

### 2. Extract Sections (`2_extract_sections.py`)

Extracts structured data from PDFs using Azure OpenAI Vision (GPT-4).

```bash
python scripts/2_extract_sections.py \
    --input data/split_pdfs/ \
    --output data/extracted/MM.csv \
    --doc-type MM \
    --workers 4
```

Supported document types: MM (Maintenance Memos), MEL, CDL, FTD

### 3. Post-process (`3_postprocess.py`)

Combines extracted fields into a single text column optimized for embeddings.

```bash
python scripts/3_postprocess.py \
    --input data/extracted/MM.csv \
    --output data/processed/MM_formatted.csv \
    --doc-type MM
```

### 4. Generate Embeddings (`4_generate_embeddings.py`)

Creates FAISS vector index from processed data.

```bash
python scripts/4_generate_embeddings.py \
    --input data/processed/MM_formatted.csv \
    --output data/vector_stores/mm_faiss_index \
    --chunk-size 2100 \
    --chunk-overlap 300
```

## Configuration

Set environment variables:

```bash
AZURE_OPENAI_ENDPOINT=<your-endpoint>
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_VISION_DEPLOYMENT=<deployment-name>
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=<deployment-name>
```

## Dependencies

```bash
pip install pdf2image pytesseract PyPDF2 Pillow pandas tqdm openai langchain langchain-openai faiss-cpu
```

System dependencies:
- Poppler (for pdf2image)
- Tesseract OCR (for text detection)

## Output

The pipeline produces:
- **CSV files**: Structured data extracted from PDFs
- **FAISS indices**: Vector embeddings for semantic search
- **Metadata**: Mappings between vectors and source documents

These outputs are used by the backend (module 2) for query processing.

## Error Handling

Failed extractions are logged separately. Check `error_files.csv` for any PDFs that couldn't be processed. Common issues:
- Invalid PDF format
- API rate limits
- OCR failures on low-quality scans

## Performance

- PDF splitting: ~50 pages/second
- Vision extraction: ~10-15 seconds per document (depends on page count)
- Embedding generation: ~1000 chunks/minute
- Total pipeline: ~2-3 minutes per document
