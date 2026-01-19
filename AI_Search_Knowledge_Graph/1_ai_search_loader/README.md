# ScholarBank to AI Search Loader

Loads ScholarBank documents into Azure AI Search indexes for knowledge graph operations and RAG retrieval.

## Overview

This script populates four AI Search indexes from Azure SQL and Blob Storage:
- **kg-docs-index** - Document metadata + summary vectors
- **kg-chunks-index** - Chunk-level content for RAG
- **kg-authors-index** - Author resolution
- **kg-departments-index** - Department resolution

## Prerequisites

- Python 3.8+
- Access to:
  - Azure SQL Database
  - Azure Blob Storage
  - Azure AI Search

## Configuration

Credentials stored in `.cred.json`:

```json
{
  "azure_sql": { "server", "username", "password", "database" },
  "azure_blob": { "storage_account", "storage_key", "container_name" },
  "asm_ai_search": { "endpoint", "api_key" }
}
```

## Usage

```bash
# Process all eligible documents
./run.sh

# Limit documents to process
./run.sh --limit 10

# Recreate indexes (use if schema changed)
./run.sh --recreate-indexes
```

## How It Works

### Processing Pipeline

```
1. Query eligible documents from SQL
   └─ sb_processed_status (stage=4, completed, validated, not in AI Search)

2. For each document:
   ├─ Fetch metadata from SQL (title, DOI, language, type, date)
   ├─ Fetch authors, departments, funders from SQL
   ├─ Download summary + embeddings from Blob Storage
   ├─ Download chunks + embeddings from Blob Storage
   └─ Upload to AI Search indexes (transactional)

3. Update SQL status (in_ai_search = 1)
```

### Index Schema

| Index | Key Field | Vector Field | Purpose |
|-------|-----------|--------------|---------|
| kg-docs-index | doc_id | summary_vector (3072d) | Document-level search |
| kg-chunks-index | chunk_id | chunk_vector (3072d) | RAG retrieval |
| kg-authors-index | author_node_id | - | Author resolution |
| kg-departments-index | department_node_id | - | Department resolution |

## Data Sources

**Azure SQL Tables:**
- `sb_processed` - Document metadata (title, DOI, language, type, publication_date)
- `sb_processed_status` - Processing status flags
- `sb_authors_to_doc` - Author information per document
- `sb_departments_to_doc` - Department information per document
- `sb_funders_to_doc` - Funder information per document

**Azure Blob Storage:**
```
data/scholarbank/{doc_id}/
├── summary/summary_{doc_id}.txt
├── summary_embeddings/summary_embedding_{doc_id}.npz
├── chunks/chk_{doc_id}_{num}.txt
└── embeddings/embed_{doc_id}_{num}.npz
```

## Document Selection Criteria

Documents must satisfy ALL conditions:
- `current_stage = 4`
- `stage_status = 'completed'`
- `in_ai_search = 0` (not yet uploaded)
- `is_validated = 1`

## Transactional Uploads

Each document upload is atomic:
1. Upload to kg-docs-index
2. Upload chunks to kg-chunks-index
3. Upload authors to kg-authors-index (deduplicated)
4. Upload departments to kg-departments-index (deduplicated)

**On failure**: Rollback all uploads for that document

## File Structure

| File | Purpose |
|------|---------|
| `run.sh` | Shell entry point |
| `main.py` | CLI and orchestration |
| `azure_clients.py` | SQL, Blob, and Search clients |
| `data_loader.py` | Data transformation and upload logic |
| `index_manager.py` | Index creation from schema |
| `aisearch_schema.json` | Index field definitions |

## Vector Search Configuration

- **Algorithm**: HNSW (Hierarchical Navigable Small World)
- **Dimensions**: 3072
- **Similarity**: Cosine
- **HNSW Parameters**: m=4, efConstruction=400, efSearch=500

## Logging

- Console: INFO level
- File: DEBUG level → `sb_load_data.log`

### Summary Output

```
============================================================
PROCESSING SUMMARY
============================================================
Total documents: 100
Successful: 98
Failed: 2
Success rate: 98.0%

Failed documents:
  - doc_id_5: Missing summary data
  - doc_id_42: Failed to upload chunks
```

## Troubleshooting

**Missing summary/chunks**
- Verify blob path structure matches expected pattern
- Check if earlier processing stages completed

**Index creation fails**
- Use `--recreate-indexes` to force recreation
- Check schema in `aisearch_schema.json`

**Partial upload detected**
- Script implements rollback on failure
- Re-run will skip successfully uploaded documents

## Schema Reference

See `aisearch_schema.json` for complete field definitions including:
- Field types (Edm.String, Edm.Boolean, Collection, etc.)
- Searchable/filterable/sortable flags
- Vector search profiles
