# Graph Builder - CosmosDB Gremlin

Builds a knowledge graph in CosmosDB Gremlin from document-concept mappings stored in Azure SQL and metadata from Azure AI Search.

## Overview

This script creates nodes (doc, author, department) and edges (TAGGED_AS, WRITES, COAUTHORS_WITH, PRODUCES, HAS_RESEARCH_INTEREST) in a Gremlin graph database.

**Target Collection:** `kg-graph-preview-dev`

## Prerequisites

- Python 3.8+
- Access to:
  - Azure CosmosDB Gremlin API
  - Azure AI Search (kg-docs-index, kg-authors-index, kg-departments-index)
  - Azure SQL Database

## Configuration

All credentials are stored in `var.json`:

```json
{
  "cosmos_db": { "endpoint", "database", "collection", "key" },
  "asm_ai_search": { "endpoint", "api_key", "doc_index_name", "author_index_name" },
  "azure_sql": { "server", "database", "username", "password" }
}
```

## Usage

```bash
# Run both parts (default)
./run.sh

# Run only document processing (Part 1)
./run.sh --retry-first

# Run only author research interests (Part 2)
./run.sh --retry-second

# Process a single document
./run.sh --doc-id <UUID>

# Limit number of documents
./run.sh --limit 10

# Enable debug logging
./run.sh --debug
```

## How It Works

### Part 1: Document Processing (7 Stages)

For each document in `asm_samples_doc2concepts`:

| Stage | Action | Data Source |
|-------|--------|-------------|
| 0 | Create doc node | AI Search `kg-docs-index` |
| 1 | Create TAGGED_AS edges (doc → concept) | SQL `asm_samples_doc2concepts` |
| 2 | Create author nodes | AI Search `kg-authors-index` |
| 3 | Create WRITES edges (author → doc) | From doc's `author_node_ids` |
| 4 | Create COAUTHORS_WITH edges | Between `author_<uuid>` authors only |
| 5 | Create department nodes | AI Search `kg-departments-index` |
| 6 | Create PRODUCES edges (dept → doc) | From doc's `department_node_ids` |

### Part 2: Author Research Interests

For each row in `asm_samples_author2concepts`:
- Creates HAS_RESEARCH_INTEREST edge (author → concept) with `count` property

## Progress Tracking

Progress is tracked in SQL table `asm_samples_progress`:

| Column | Type | Description |
|--------|------|-------------|
| row_id | NVARCHAR(100) | Document UUID |
| current_stage | INT | Current stage (0-6) |
| stage_status | VARCHAR(20) | `pending`, `in-progress`, `completed`, `errored` |
| retry_count | INT | Retry attempts (max 3) |
| err_message | NVARCHAR(500) | Error details |

The script automatically resumes from the last successful stage.

## Data Flow

```
SQL: asm_samples_doc2concepts
         │
         ▼
    unique doc_ids (518)
         │
         ▼
    For each doc_id:
    ┌────────────────────────────────────────┐
    │ Stage 0: doc node (from AI Search)     │
    │ Stage 1: TAGGED_AS edges               │
    │ Stage 2: author nodes (from AI Search) │
    │ Stage 3: WRITES edges                  │
    │ Stage 4: COAUTHORS_WITH edges          │
    │ Stage 5: department nodes              │
    │ Stage 6: PRODUCES edges                │
    └────────────────────────────────────────┘
         │
         ▼
SQL: asm_samples_author2concepts
         │
         ▼
    HAS_RESEARCH_INTEREST edges (author → concept)
```

## Key Design Decisions

1. **Idempotent**: All node/edge creations check existence first - safe to re-run
2. **COAUTHORS_WITH**: Only between `author_<uuid>` format authors (not `author_appearance_*`)
3. **Single direction**: COAUTHORS_WITH uses alphabetical ordering (lower → higher ID)
4. **Retry logic**: Max 3 retries per stage before marking as permanently errored

## File Structure

| File | Purpose |
|------|---------|
| `run.sh` | Shell entry point (creates venv, installs deps) |
| `main.py` | CLI entry point |
| `config.py` | Load credentials, setup logging |
| `gremlin_client.py` | Gremlin node/edge operations |
| `search_client.py` | AI Search lookups |
| `sql_client.py` | SQL data retrieval + progress tracking |
| `graph_builder.py` | Part 1: Document processing stages |
| `author_interest_builder.py` | Part 2: Author research interests |

## Logging

- Console: INFO level
- File: DEBUG level → `graph_builder.log`

## Troubleshooting

**No documents processed (0 total)**
- Check if `asm_samples_progress` table has NULL values in `current_stage`/`stage_status`
- The query uses COALESCE to handle NULLs as stage 0 with 'pending' status

**Document stuck at a stage**
- Check `err_message` in `asm_samples_progress`
- If `retry_count >= 3`, manually reset to retry:
  ```sql
  UPDATE asm_samples_progress
  SET retry_count = 0, stage_status = 'pending'
  WHERE row_id = '<doc_id>'
  ```

**Edge creation fails**
- Verify target node exists (e.g., concept node for TAGGED_AS)
- Check Gremlin query escaping for special characters in titles

## Graph Schema Reference

See `graph_schema.json` for complete node/edge property definitions.
