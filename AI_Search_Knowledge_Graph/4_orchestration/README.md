# End-to-End Workflow

This folder contains a complete end-to-end workflow for processing, indexing, and building a knowledge graph from document data. The workflow integrates the functionalities of the `sampletoconceptmapping`, `sb_load_ai_search`, and `sb_load_graph_db` modules to provide a seamless pipeline for data processing and knowledge graph creation.

## Workflow Overview

1. **Data Loading**:
   - Load documents into Azure AI Search indexes using the `sb_load_ai_search` module.
   - This step ensures that the data is indexed and ready for hybrid search.

2. **Document-to-Concept Mapping**:
   - Map documents to relevant concepts using hybrid search and LLM evaluation with the `sampletoconceptmapping` module.
   - Store the results in Azure SQL for downstream processing.

3. **Graph Database Building**:
   - Build a knowledge graph from the document-concept mappings and metadata using the `sb_load_graph_db` module.
   - This step creates nodes and edges in CosmosDB Gremlin for advanced querying.

## Prerequisites

- Python 3.8+
- Access to:
  - Azure AI Search
  - Azure SQL Database
  - Azure CosmosDB Gremlin API

## Usage

To execute the entire workflow, run the following command:

```bash
./run.sh
```

This will execute the following steps in sequence:
1. Load data into Azure AI Search indexes.
2. Map documents to concepts using hybrid search and LLM evaluation.
3. Build a knowledge graph in CosmosDB Gremlin.

### Custom Execution

You can also execute individual steps of the workflow by running the corresponding scripts in the respective folders:

1. **Data Loading**:
   ```bash
   cd ../sb_load_ai_search
   ./run.sh
   ```

2. **Document-to-Concept Mapping**:
   ```bash
   cd ../sampletoconceptmapping
   ./run.sh
   ```

3. **Graph Database Building**:
   ```bash
   cd ../sb_load_graph_db
   ./run.sh
   ```

## File Structure

| File | Purpose |
|------|---------|
| `run.sh` | Orchestrates the entire end-to-end workflow |
| `README.md` | Documentation for the workflow |

## Notes

- Ensure that all dependencies are installed by running `pip install -r requirements.txt` in each module folder.
- Update the configuration files (`var.json`, `.cred.json`) in the respective folders with the correct credentials and settings before running the workflow.