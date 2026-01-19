# Document-to-Concept Mapping

Maps ScholarBank documents to knowledge graph concepts using hybrid search and LLM-as-Judge evaluation.

## Overview

This script automatically maps documents to relevant concepts by:
1. Retrieving document metadata from Azure AI Search
2. Performing hybrid search (BM25 + vector) to find candidate concepts
3. Using an LLM judge to score and rank concept relevance
4. Storing results in Azure SQL for downstream graph building

## Prerequisites

- Python 3.8+
- Access to:
  - Azure AI Search (kg-docs-index, kg-concepts-index)
  - Azure SQL Database
  - Azure Foundry (GPT-4.1)

## Configuration

Credentials stored in `var.json`:

```json
{
  "azure_foundry": { "endpoint", "deployment_name", "api_key", "api_version" },
  "asm_ai_search": { "endpoint", "api_key", "doc_index_name", "concept_index_name" },
  "azure_sql": { "server", "database", "username", "password" }
}
```

## Usage

```bash
# Process all eligible documents
./run.sh

# Limit documents to process
./run.sh --limit 10

# Process a single document
./run.sh --doc-id <UUID>

# Backfill author2concepts from existing doc2concepts
./run.sh --fill-authors

# Enable debug logging
./run.sh --debug
```

## How It Works

### Processing Pipeline

```
1. Query eligible documents
   └─ sb_processed_status (stage=4, completed, validated, in_ai_search)

2. For each document:
   ├─ Retrieve from kg-docs-index (title, summary, authors)
   ├─ Hybrid search kg-concepts-index → 50 candidates
   ├─ LLM judge evaluates and scores → top 20
   └─ Store top 15 in SQL (transactional)

3. Output:
   ├─ asm_samples_doc2concepts (doc_id, concept_id, strength)
   └─ asm_samples_author2concepts (author_id, concept_id, num_doc)
```

### LLM Judge Retry Strategy

| Attempt | Strategy |
|---------|----------|
| 1-2 | Original prompt |
| 3-5 | Modified prompt emphasizing distinct concepts |
| Fallback | Consolidate results from all attempts |

Minimum 15 valid concepts required to accept results.

## Data Flow

**Sources:**
- Azure AI Search `kg-docs-index` - Document metadata
- Azure AI Search `kg-concepts-index` - Concept candidates
- Azure SQL `sb_processed_status` - Eligible documents

**Destinations:**
- Azure SQL `asm_samples_doc2concepts` - Document-concept mappings with strength (0-10)
- Azure SQL `asm_samples_author2concepts` - Author-concept mappings with document count

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hybrid_search_top_k` | 50 | Initial concept candidates |
| `llm_judge_top_k` | 20 | Concepts returned by LLM |
| `final_top_k` | 15 | Concepts stored per document |
| `max_retries` | 5 | LLM judge retry attempts |
| `llm_timeout` | 300s | LLM response timeout |

## Progress Tracking

- **Resumable**: Skips documents already in `asm_samples_doc2concepts`
- **Transactional**: All-or-nothing SQL inserts prevent partial updates
- **Failed docs**: Logged to `failed_docs.json` with error details

### Summary Output

```
PROCESSING SUMMARY
==================================================
Total eligible documents: 1000
Already processed (skipped): 950
Processed this run: 45
Failed this run: 5
Success rate: 90.0%
Duration: 1234.56s
==================================================
```

## File Structure

| File | Purpose |
|------|---------|
| `run.sh` | Shell entry point |
| `main.py` | CLI and orchestration |
| `config.py` | Configuration loader |
| `sql_client.py` | SQL operations (query, insert, upsert) |
| `search_client.py` | AI Search hybrid queries |
| `llm_judge.py` | LLM concept evaluation with retry |

## Logging

- Console: INFO level
- File: DEBUG level → `processing.log`

## Author Backfill Mode

The `--fill-authors` flag populates `asm_samples_author2concepts` from existing `asm_samples_doc2concepts`:

```bash
./run.sh --fill-authors --limit 100
```

For each document:
1. Get authors from AI Search (only `author_<UUID>` format)
2. Get concepts from `asm_samples_doc2concepts`
3. UPSERT into `asm_samples_author2concepts` (increments `num_doc`)

## Troubleshooting

**LLM returns too few concepts**
- Check LLM response in debug logs
- May need to adjust prompt or retry parameters

**SQL truncation error**
- Verify column widths match data (doc_id should be NVARCHAR(100))

**Document not found in AI Search**
- Ensure document was loaded via `sb_load_data_to_aisearch` first
