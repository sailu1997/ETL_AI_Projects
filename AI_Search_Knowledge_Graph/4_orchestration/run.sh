#!/bin/bash

# End-to-End Workflow Script
# This script orchestrates the entire workflow: data loading, document-to-concept mapping, and graph database building.

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Data Loading
cd ../sb_load_ai_search
./run.sh

# Step 2: Document-to-Concept Mapping
cd ../sampletoconceptmapping
./run.sh

# Step 3: Graph Database Building
cd ../sb_load_graph_db
./run.sh

# Return to the end_to_end_workflow directory
cd ../end_to_end_workflow

echo "End-to-end workflow completed successfully!"