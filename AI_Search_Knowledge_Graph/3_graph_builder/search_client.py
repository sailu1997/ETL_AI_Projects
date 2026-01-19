"""Azure AI Search client for retrieving document and author data."""
import logging
from typing import Dict, Optional, Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient as AzureSearchClient
from azure.core.exceptions import ResourceNotFoundError

from config import config

logger = logging.getLogger(__name__)


class SearchClient:
    """Client for Azure AI Search operations."""

    def __init__(self):
        self.credential = AzureKeyCredential(config.search_api_key)

        # Initialize clients for different indexes
        self._doc_client = AzureSearchClient(
            endpoint=config.search_endpoint,
            index_name=config.doc_index,
            credential=self.credential
        )
        self._author_client = AzureSearchClient(
            endpoint=config.search_endpoint,
            index_name=config.author_index,
            credential=self.credential
        )
        self._department_client = AzureSearchClient(
            endpoint=config.search_endpoint,
            index_name=config.department_index,
            credential=self.credential
        )
        logger.info("AI Search clients initialized")

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve document from kg-docs-index by doc_id.

        Args:
            doc_id: Document ID (format: doc_<uuid>) - this is the key in kg-docs-index

        Returns:
            Dict with title, source, hidden, author_node_ids, department_node_ids
            or None if not found
        """
        try:
            result = self._doc_client.get_document(
                key=doc_id,
                selected_fields=[
                    'doc_id', 'node_id', 'title', 'source', 'hidden',
                    'author_node_ids', 'department_node_ids'
                ]
            )
            return dict(result)
        except ResourceNotFoundError:
            logger.warning(f"Document not found in AI Search: {doc_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve document {doc_id}: {e}")
            return None

    def get_author(self, author_node_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve author from kg-authors-index.

        Args:
            author_node_id: Author node ID (format: author_<uuid> or author_appearance_<...>)
                            This is the key (author_node_id) in kg-authors-index

        Returns:
            Dict with author_id, normalized_name, display_name, type
            or None if not found
        """
        try:
            result = self._author_client.get_document(
                key=author_node_id,
                selected_fields=[
                    'author_node_id', 'author_id', 'normalized_name',
                    'display_name', 'type'
                ]
            )
            return dict(result)
        except ResourceNotFoundError:
            logger.debug(f"Author not found in AI Search index: {author_node_id}")
            return None
        except Exception as e:
            logger.debug(f"Failed to retrieve author {author_node_id}: {e}")
            return None

    def get_department(self, department_node_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve department from kg-departments-index.

        Args:
            department_node_id: Department node ID (format: department_<id>)
                                This is the key (department_node_id) in kg-departments-index

        Returns:
            Dict with department_id, normalized_name, display_name
            or None if not found
        """
        try:
            result = self._department_client.get_document(
                key=department_node_id,
                selected_fields=[
                    'department_node_id', 'department_id',
                    'normalized_name', 'display_name'
                ]
            )
            return dict(result)
        except ResourceNotFoundError:
            logger.debug(f"Department not found in AI Search index: {department_node_id}")
            return None
        except Exception as e:
            logger.debug(f"Failed to retrieve department {department_node_id}: {e}")
            return None
