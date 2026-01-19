# AI Search + Knowledge Graph System

**Intelligent Document Processing Pipeline with Azure AI Search and Graph Databases**

A production-ready system demonstrating hybrid search, knowledge graph construction, and LLM-powered concept mapping using Azure cloud services.

---

## Project Overview

This system processes academic documents through a **4-stage pipeline** that combines:
- **Azure AI Search** (vector + keyword hybrid search)
- **Azure Cosmos DB** (Gremlin graph database)
- **LLM-as-Judge** pattern for intelligent concept mapping
- **Multi-stage ETL orchestration**

### Business Value

- **Enhanced Discoverability**: Documents are searchable via semantic embeddings and metadata
- **Relationship Insights**: Graph database reveals author collaborations, research themes, and institutional connections
- **Intelligent Categorization**: LLM judges automatically tag documents with relevant concepts
- **Scalable Architecture**: Handles large document corpuses with resumable, transactional processing

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Azure SQL Database                         │
│         (Document Metadata + Processing Status)              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│     STAGE 1: AI Search Loader                               │
│     ├─ Load documents + embeddings                          │
│     ├─ Create 4 search indexes (docs, chunks, authors, depts)│
│     └─ HNSW vector search (cosine, 3072d)                   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│     STAGE 2: Concept Mapper                                 │
│     ├─ Hybrid search (BM25 + vector) → 50 candidates       │
│     ├─ LLM judge evaluates relevance → top 20              │
│     └─ Store top 15 concepts per document                  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│     STAGE 3: Graph Builder                                  │
│     ├─ Create nodes (doc, author, dept, concept)           │
│     ├─ Create edges (TAGGED_AS, WRITES, COAUTHORS_WITH,    │
│     │                PRODUCES, HAS_RESEARCH_INTEREST)       │
│                   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│     STAGE 4: Orchestration                                  │
│     └─ End-to-end workflow execution                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
AI_Search_Knowledge_Graph/
│
├── README.md                           # This file
├── .gitignore                          # Security & cleanup
│
├── 1_ai_search_loader/                 # STAGE 1: Data Indexing
│   ├── README.md                       # Detailed documentation
│   ├── requirements.txt
│   ├── env_template.txt                # Configuration template
│   ├── run.sh                          # Execution script
│   ├── main.py                         # CLI orchestration
│   ├── azure_clients.py                # SQL, Blob, Search clients
│   ├── data_loader.py                  # Transform & upload logic
│   ├── index_manager.py                # Index creation
│   └── aisearch_schema.json            # Index field definitions
│
├── 2_concept_mapper/                   # STAGE 2: LLM Concept Mapping
│   ├── README.md
│   ├── requirements.txt
│   ├── env_template.txt
│   ├── run.sh
│   ├── main.py                         # CLI orchestration
│   ├── config.py                       # Config loader
│   ├── search_client.py                # Hybrid search
│   ├── llm_judge.py                    # LLM evaluation with retry
│   ├── sql_client.py                   # SQL operations
│   └── failed_docs.json                # Error tracking (generated)
│
├── 3_graph_builder/                    # STAGE 3: Knowledge Graph
│   ├── README.md
│   ├── requirements.txt
│   ├── env_template.txt
│   ├── run.sh
│   ├── main.py                         # CLI orchestration
│   ├── config.py                       # Config loader
│   ├── gremlin_client.py               # Gremlin operations
│   ├── search_client.py                # AI Search lookups
│   ├── sql_client.py                   # Progress tracking
│   ├── graph_builder.py                # Document processing (7 stages)
│   ├── author_interest_builder.py      # Author-concept mapping
│   └── graph_schema.json               # Node/edge definitions
│
└── 4_orchestration/                    # STAGE 4: End-to-End Workflow
    ├── README.md
    └── run.sh                          # Execute all stages
```

---

## Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Azure Resources**:
   - Azure SQL Database
   - Azure Blob Storage
   - Azure AI Search
   - Azure Cosmos DB (Gremlin API)
   - Azure OpenAI / Foundry (GPT-4)

### Installation

```bash
# Clone or navigate to project
cd AI_Search_Knowledge_Graph

# Each module has its own environment
cd 1_ai_search_loader
pip install -r requirements.txt

# Configure credentials (repeat for each stage)
cp env_template.txt .env
# Edit .env with your Azure credentials
```

### Execution

#### Option 1: Run Individual Stages

```bash
# Stage 1: Load to AI Search
cd 1_ai_search_loader
./run.sh --limit 100

# Stage 2: Map Concepts
cd ../2_concept_mapper
./run.sh --limit 100

# Stage 3: Build Graph
cd ../3_graph_builder
./run.sh --limit 100
```

#### Option 2: Run Complete Workflow

```bash
cd 4_orchestration
./run.sh
```

---

## Technical Highlights

### 1. Hybrid Search (AI Search)

- **BM25**: Traditional keyword matching
- **Vector Search**: 3072-dimensional embeddings (HNSW algorithm)
- **Combined Scoring**: Weighted sum for optimal retrieval

### 2. LLM-as-Judge Pattern

- **Retry Strategy**: 5 attempts with modified prompts
- **Quality Control**: Requires minimum 15 valid concepts
- **Fallback Logic**: Consolidates results from multiple attempts

### 3. Graph Database Schema

**Nodes:**
- `doc` (document_id, title, DOI, publication_date)
- `author` (author_node_id, name, affiliation)
- `department` (department_node_id, name)
- `concept` (concept_id, name, description)

**Edges:**
- `TAGGED_AS` (doc → concept, strength: 0-10)
- `WRITES` (author → doc)
- `COAUTHORS_WITH` (author ↔ author)
- `PRODUCES` (department → doc)
- `HAS_RESEARCH_INTEREST` (author → concept, count)

### 4. Transactional Processing

- **Atomic Operations**: All-or-nothing inserts
- **Progress Tracking**: Resume from last successful stage
- **Retry Logic**: Max 3 retries per stage with exponential backoff
- **Idempotent**: Safe to re-run without duplication

### 5. Data Quality

- **Validation**: Document metadata completeness checks
- **Deduplication**: Author/department nodes merged by ID
- **Error Handling**: Failed documents logged for manual review

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Document Processing** | ~5-10 docs/minute |
| **Concept Mapping** | ~30s per document (LLM latency) |
| **Graph Operations** | ~2-3 minutes per document (7 stages) |
| **Vector Dimensions** | 3072 (OpenAI text-embedding-3-large) |
| **Hybrid Search Candidates** | 50 concepts |
| **Final Concepts per Doc** | 15 (top-scored) |

---

## Configuration

Each stage requires a configuration file:

### `1_ai_search_loader/.cred.json`

```json
{
  "azure_sql": {
    "server": "your-server.database.windows.net",
    "database": "your-database",
    "username": "your-username",
    "password": "your-password"
  },
  "azure_blob": {
    "storage_account": "your-storage-account",
    "storage_key": "your-key",
    "container_name": "your-container"
  },
  "asm_ai_search": {
    "endpoint": "https://your-search.search.windows.net",
    "api_key": "your-api-key"
  }
}
```

### `2_concept_mapper/var.json` & `3_graph_builder/var.json`

```json
{
  "azure_foundry": {
    "endpoint": "https://your-openai.openai.azure.com",
    "deployment_name": "gpt-4",
    "api_key": "your-api-key",
    "api_version": "2024-02-15-preview"
  },
  "asm_ai_search": {
    "endpoint": "https://your-search.search.windows.net",
    "api_key": "your-api-key",
    "doc_index_name": "kg-docs-index",
    "concept_index_name": "kg-concepts-index"
  },
  "azure_sql": {
    "server": "your-server.database.windows.net",
    "database": "your-database",
    "username": "your-username",
    "password": "your-password"
  },
  "cosmos_db": {
    "endpoint": "wss://your-cosmosdb.gremlin.cosmos.azure.com:443/",
    "database": "your-database",
    "collection": "kg-graph-preview-dev",
    "key": "your-key"
  }
}
```

---

## Troubleshooting

### Stage 1: AI Search Loader

**Issue**: Missing summary/chunks in Blob Storage
```bash
# Verify blob path structure
az storage blob list --account-name <account> --container-name <container> --prefix data/scholarbank/
```

**Issue**: Index creation fails
```bash
# Recreate indexes
./run.sh --recreate-indexes
```

### Stage 2: Concept Mapper

**Issue**: LLM returns too few concepts
- Check `processing.log` for LLM responses
- May need to adjust `llm_judge_top_k` in config

**Issue**: Document not found in AI Search
- Ensure Stage 1 completed successfully
- Check `in_ai_search` flag in `sb_processed_status` table

### Stage 3: Graph Builder

**Issue**: Document stuck at a stage
```sql
-- Check error message
SELECT row_id, current_stage, stage_status, retry_count, err_message
FROM asm_samples_progress
WHERE stage_status = 'errored';

-- Reset to retry
UPDATE asm_samples_progress
SET retry_count = 0, stage_status = 'pending'
WHERE row_id = '<doc_id>';
```


## Use Cases

### 1. Research Trend Analysis
```gremlin
// Find top research concepts
g.V().hasLabel('concept').inE('TAGGED_AS').count().order().by(decr).limit(10)
```

### 2. Author Collaboration Networks
```gremlin
// Find author collaboration clusters
g.V().hasLabel('author').outE('COAUTHORS_WITH').inV().path()
```

### 3. Institutional Research Profiles
```gremlin
// Department research strengths
g.V().hasLabel('department').out('PRODUCES').out('TAGGED_AS').groupCount()
```

### 4. Semantic Document Search
```python
# Hybrid search via AI Search API
results = search_client.search(
    search_text="machine learning",
    vector_queries=[VectorQuery(
        vector=query_embedding,
        k_nearest_neighbors=10,
        fields="summary_vector"
    )]
)
```

---



## Technology Stack

| Category | Technology |
|----------|-----------|
| **Cloud Platform** | Microsoft Azure |
| **Search** | Azure AI Search (Cognitive Search) |
| **Database** | Azure SQL Database |
| **Graph DB** | Azure Cosmos DB (Gremlin API) |
| **Storage** | Azure Blob Storage |
| **LLM** | Azure OpenAI / Foundry (GPT-4) |
| **Vector Algorithm** | HNSW (Hierarchical Navigable Small World) |
| **Embedding Model** | OpenAI text-embedding-3-large (3072d) |
| **Programming** | Python 3.9+ |
| **Libraries** | azure-search-documents, gremlinpython, pyodbc, openai |







