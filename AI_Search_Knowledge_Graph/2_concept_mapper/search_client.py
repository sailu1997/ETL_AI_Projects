"""Azure AI Search operations for concept mapping."""
import logging
import re
from typing import List, Dict, Optional, Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient as AzureSearchClient
from azure.search.documents.models import VectorizedQuery

from config import config

logger = logging.getLogger(__name__)


class SearchClient:
    """Client for Azure AI Search operations."""

    def __init__(self):
        self.endpoint = config.search_endpoint
        self.credential = AzureKeyCredential(config.search_api_key)

        # Initialize clients for different indexes
        self._doc_client = AzureSearchClient(
            endpoint=self.endpoint,
            index_name=config.doc_index,  # kg-docs-index
            credential=self.credential
        )
        self._concept_client = AzureSearchClient(
            endpoint=self.endpoint,
            index_name=config.concept_index,  # kg-concepts-index
            credential=self.credential
        )
        logger.info(f"Search clients initialized for {config.doc_index} and {config.concept_index}")

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve document from kg-docs-index by doc_id.

        Args:
            doc_id: The document row_id (without prefix)

        Returns:
            Dict with title, summary, summary_vector, author_node_ids
            or None if not found
        """
        try:
            # doc_id in index is prefixed: doc_<row_id>
            prefixed_id = f"doc_{doc_id}"
            result = self._doc_client.get_document(
                key=prefixed_id,
                selected_fields=['doc_id', 'title', 'summary', 'summary_vector', 'author_node_ids']
            )
            return dict(result)
        except Exception as e:
            logger.error(f"Failed to retrieve document {doc_id}: {e}")
            return None

    def filter_author_node_ids(self, author_node_ids: List[str]) -> List[str]:
        """
        Filter author_node_ids to only include 'author_xxxx' pattern.
        Exclude 'author_appearance_xxxx' entries.

        Args:
            author_node_ids: List of author node IDs from the document

        Returns:
            Filtered list with only 'author_xxxx' pattern
        """
        if not author_node_ids:
            return []

        # Pattern matches author_<UUID> but not author_appearance_*
        # UUID format: 36 chars with hyphens (e.g., author_97bc05d3-48cc-4cbf-afd1-1313e3e552e0)
        pattern = re.compile(r'^author_[a-f0-9-]{36}$')
        filtered = [aid for aid in author_node_ids if pattern.match(aid)]
        logger.debug(f"Filtered {len(author_node_ids)} -> {len(filtered)} author IDs")
        return filtered

    def search_concepts_hybrid(
        self,
        query_text: str,
        embedding: List[float],
        top_k: int = 50
    ) -> List[Dict]:
        """
        Hybrid search (text BM25 + vector similarity) against kg-concepts-index.

        Args:
            query_text: Text for BM25 search (document summary)
            embedding: 3072-dim vector for similarity search
            top_k: Number of results to retrieve

        Returns:
            List of {concept_id, label, description, score, rank}
        """
        try:
            # Build vector query
            vector_query = VectorizedQuery(
                vector=embedding,
                k_nearest_neighbors=top_k,
                fields="concept_description_vector"
            )

            # Perform hybrid search
            results = self._concept_client.search(
                search_text=query_text[:1000],  # Limit query text length
                vector_queries=[vector_query],
                select=["concept_id", "concept_label", "concept_description"],
                top=top_k,
                include_total_count=True
            )

            search_results = []
            for idx, result in enumerate(results):
                search_results.append({
                    'concept_id': result.get('concept_id', ''),
                    'label': result.get('concept_label', ''),
                    'description': result.get('concept_description', ''),
                    'score': result.get('@search.score', 0.0),
                    'rank': idx + 1
                })

            logger.info(f"Hybrid search returned {len(search_results)} concepts")
            return search_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
