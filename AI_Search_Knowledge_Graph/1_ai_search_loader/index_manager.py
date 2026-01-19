"""AI Search index management - create indexes from schema if they don't exist."""

import json
import logging
from pathlib import Path
from typing import Dict, List

from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)

logger = logging.getLogger(__name__)

# Indexes to create (skip kg-concepts-index)
INDEXES_TO_CREATE = [
    'kg-docs-index',
    'kg-chunks-index',
    'kg-authors-index',
    'kg-departments-index'
]


def load_schema(schema_path: Path) -> Dict:
    """Load the AI Search schema from JSON file."""
    with open(schema_path, 'r') as f:
        return json.load(f)


def map_field_type(edm_type: str) -> SearchFieldDataType:
    """Map EDM type string to SearchFieldDataType."""
    type_mapping = {
        'Edm.String': SearchFieldDataType.String,
        'Edm.Boolean': SearchFieldDataType.Boolean,
        'Edm.DateTimeOffset': SearchFieldDataType.DateTimeOffset,
        'Edm.Int32': SearchFieldDataType.Int32,
        'Edm.Int64': SearchFieldDataType.Int64,
        'Edm.Double': SearchFieldDataType.Double,
        'Collection(Edm.String)': SearchFieldDataType.Collection(SearchFieldDataType.String),
        'Collection(Edm.Single)': SearchFieldDataType.Collection(SearchFieldDataType.Single),
    }
    return type_mapping.get(edm_type, SearchFieldDataType.String)


def create_search_field(field_def: Dict, vector_profiles: Dict[str, str]) -> SearchField:
    """Create a SearchField from field definition."""
    field_type = field_def.get('type', 'Edm.String')
    is_vector = 'vectorSearchDimensions' in field_def
    is_collection = field_type.startswith('Collection(')

    if is_vector:
        # Vector field
        return SearchField(
            name=field_def['name'],
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=field_def['vectorSearchDimensions'],
            vector_search_profile_name=field_def.get('vectorSearchProfile', 'defaultVectorProfile')
        )
    elif field_def.get('key', False):
        # Key field
        return SimpleField(
            name=field_def['name'],
            type=map_field_type(field_type),
            key=True,
            filterable=field_def.get('filterable', True),
            sortable=field_def.get('sortable', False)
        )
    elif is_collection:
        # Collection field - must use SearchField directly, not SearchableField
        return SearchField(
            name=field_def['name'],
            type=map_field_type(field_type),
            searchable=field_def.get('searchable', False),
            filterable=field_def.get('filterable', False),
            sortable=field_def.get('sortable', False),
            facetable=field_def.get('facetable', False)
        )
    elif field_def.get('searchable', False):
        # Searchable scalar field
        return SearchableField(
            name=field_def['name'],
            type=map_field_type(field_type),
            filterable=field_def.get('filterable', False),
            sortable=field_def.get('sortable', False),
            facetable=field_def.get('facetable', False)
        )
    else:
        # Simple scalar field
        return SimpleField(
            name=field_def['name'],
            type=map_field_type(field_type),
            filterable=field_def.get('filterable', False),
            sortable=field_def.get('sortable', False),
            facetable=field_def.get('facetable', False)
        )


def get_vector_search_config(index_name: str) -> VectorSearch:
    """Get vector search configuration for an index."""
    # Define HNSW algorithm configurations
    algorithms = [
        HnswAlgorithmConfiguration(
            name="hnswConfig",
            parameters={
                "m": 4,
                "efConstruction": 400,
                "efSearch": 500,
                "metric": "cosine"
            }
        )
    ]

    # Define profiles based on index
    profiles = []
    if index_name == 'kg-docs-index':
        profiles.append(VectorSearchProfile(
            name="summaryVectorProfile",
            algorithm_configuration_name="hnswConfig"
        ))
    elif index_name == 'kg-chunks-index':
        profiles.append(VectorSearchProfile(
            name="chunkVectorProfile",
            algorithm_configuration_name="hnswConfig"
        ))
    elif index_name == 'kg-concepts-index':
        profiles.append(VectorSearchProfile(
            name="conceptVectorProfile",
            algorithm_configuration_name="hnswConfig"
        ))

    # Add a default profile for any index
    profiles.append(VectorSearchProfile(
        name="defaultVectorProfile",
        algorithm_configuration_name="hnswConfig"
    ))

    return VectorSearch(
        algorithms=algorithms,
        profiles=profiles
    )


def delete_index(index_client: SearchIndexClient, index_name: str) -> bool:
    """Delete an index if it exists.

    Returns True if index was deleted or didn't exist.
    """
    try:
        existing_indexes = [idx.name for idx in index_client.list_indexes()]
        if index_name in existing_indexes:
            index_client.delete_index(index_name)
            logger.info(f"Deleted index: {index_name}")
        return True
    except Exception as e:
        logger.error(f"Error deleting index {index_name}: {e}")
        return False


def create_index_from_schema(
    index_client: SearchIndexClient,
    index_name: str,
    schema: Dict,
    force_recreate: bool = False
) -> bool:
    """Create an index from schema definition.

    Returns True if index was created or already exists.
    """
    try:
        # Check if index already exists
        existing_indexes = [idx.name for idx in index_client.list_indexes()]
        if index_name in existing_indexes:
            if force_recreate:
                logger.info(f"Recreating index '{index_name}'...")
                if not delete_index(index_client, index_name):
                    return False
            else:
                logger.info(f"Index '{index_name}' already exists, skipping creation")
                return True

        # Get schema for this index
        index_schema = schema.get('searchIndexSchemas', {}).get(index_name)
        if not index_schema:
            logger.error(f"Schema not found for index: {index_name}")
            return False

        # Create fields
        vector_profiles = {}  # Track vector profile names
        fields = []
        for field_def in index_schema.get('fields', []):
            field = create_search_field(field_def, vector_profiles)
            fields.append(field)

        # Create index with vector search config
        vector_search = get_vector_search_config(index_name)

        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search
        )

        # Create the index
        index_client.create_or_update_index(index)
        logger.info(f"Successfully created index: {index_name}")
        return True

    except Exception as e:
        logger.error(f"Error creating index {index_name}: {e}")
        return False


def ensure_indexes_exist(
    index_client: SearchIndexClient,
    schema_path: Path,
    force_recreate: bool = False
) -> bool:
    """Ensure all required indexes exist.

    Args:
        index_client: The SearchIndexClient to use
        schema_path: Path to the schema JSON file
        force_recreate: If True, delete and recreate existing indexes

    Returns True if all indexes are ready.
    """
    schema = load_schema(schema_path)

    all_success = True
    for index_name in INDEXES_TO_CREATE:
        success = create_index_from_schema(index_client, index_name, schema, force_recreate)
        if not success:
            all_success = False
            logger.error(f"Failed to ensure index: {index_name}")

    return all_success
