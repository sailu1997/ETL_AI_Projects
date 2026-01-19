"""Azure client handlers for SQL, Blob Storage, and AI Search."""

import logging
from typing import Dict, List, Optional, Any
from io import BytesIO

import pymssql
import numpy as np
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

logger = logging.getLogger(__name__)


class SQLClient:
    """Client for Azure SQL database operations."""

    def __init__(self, config: Dict[str, str]):
        self.server = config['server']
        self.username = config['username']
        self.password = config['password']
        self.database = config['database']

    def _get_connection(self):
        """Create a new database connection."""
        return pymssql.connect(
            server=self.server,
            user=self.username,
            password=self.password,
            database=self.database,
            tds_version='7.4',
            timeout=30
        )

    def get_eligible_documents(self, limit: Optional[int] = None) -> List[Dict]:
        """Get documents eligible for AI Search loading.

        Criteria: current_stage=4, stage_status=completed, in_ai_search=False, is_validated=True
        """
        query = """
            SELECT TOP {limit} row_id
            FROM dbo.sb_processed_status
            WHERE current_stage = 4
              AND stage_status = 'completed'
              AND in_ai_search = 0
              AND is_validated = 1
        """.format(limit=limit if limit else 1000000)

        with self._get_connection() as conn:
            cursor = conn.cursor(as_dict=True)
            cursor.execute(query)
            return cursor.fetchall()

    def get_document_metadata(self, doc_id: str) -> Optional[Dict]:
        """Get document metadata from sb_processed table."""
        query = """
            SELECT id, doc_type, published_date, doi, doc_uri, title, language
            FROM dbo.sb_processed
            WHERE id = %s
        """
        with self._get_connection() as conn:
            cursor = conn.cursor(as_dict=True)
            cursor.execute(query, (doc_id,))
            result = cursor.fetchone()
            return result

    def get_authors(self, doc_id: str) -> List[Dict]:
        """Get authors for a document from sb_authors_to_doc table."""
        query = """
            SELECT doc_id, author_id, normalized_name, display_name, place, author_type
            FROM dbo.sb_authors_to_doc
            WHERE doc_id = %s
            ORDER BY place
        """
        with self._get_connection() as conn:
            cursor = conn.cursor(as_dict=True)
            cursor.execute(query, (doc_id,))
            return cursor.fetchall()

    def get_departments(self, doc_id: str) -> List[Dict]:
        """Get departments for a document from sb_departments_to_doc table."""
        query = """
            SELECT doc_id, department_id, normalized_name, display_name
            FROM dbo.sb_departments_to_doc
            WHERE doc_id = %s
        """
        with self._get_connection() as conn:
            cursor = conn.cursor(as_dict=True)
            cursor.execute(query, (doc_id,))
            return cursor.fetchall()

    def get_funders(self, doc_id: str) -> List[Dict]:
        """Get funders for a document from sb_funders_to_doc table."""
        query = """
            SELECT doc_id, normalized_name, display_name
            FROM dbo.sb_funders_to_doc
            WHERE doc_id = %s
        """
        with self._get_connection() as conn:
            cursor = conn.cursor(as_dict=True)
            cursor.execute(query, (doc_id,))
            return cursor.fetchall()

    def update_ai_search_status(self, doc_id: str, status: bool) -> None:
        """Update in_ai_search status for a document."""
        query = """
            UPDATE dbo.sb_processed_status
            SET in_ai_search = %s
            WHERE row_id = %s
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (1 if status else 0, doc_id))
            conn.commit()
            logger.debug(f"Updated in_ai_search={status} for doc_id={doc_id}")


class BlobClient:
    """Client for Azure Blob Storage operations."""

    def __init__(self, config: Dict[str, str]):
        connection_string = (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={config['storage_account']};"
            f"AccountKey={config['storage_key']};"
            f"EndpointSuffix=core.windows.net"
        )
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_name = config['container_name']
        self.container_client = self.blob_service_client.get_container_client(self.container_name)

    def download_text(self, blob_path: str) -> Optional[str]:
        """Download text content from a blob."""
        try:
            blob_client = self.container_client.get_blob_client(blob_path)
            if not blob_client.exists():
                logger.warning(f"Blob not found: {blob_path}")
                return None
            return blob_client.download_blob().readall().decode('utf-8')
        except Exception as e:
            logger.error(f"Error downloading text blob {blob_path}: {e}")
            return None

    def download_embedding(self, blob_path: str) -> Optional[List[float]]:
        """Download embedding from .npz file."""
        try:
            blob_client = self.container_client.get_blob_client(blob_path)
            if not blob_client.exists():
                logger.warning(f"Blob not found: {blob_path}")
                return None

            data = blob_client.download_blob().readall()
            npz_file = np.load(BytesIO(data))
            # Assume the embedding is stored under the first key or 'embedding' key
            keys = list(npz_file.keys())
            if 'embedding' in keys:
                embedding = npz_file['embedding']
            elif len(keys) > 0:
                embedding = npz_file[keys[0]]
            else:
                logger.error(f"No data found in npz file: {blob_path}")
                return None

            return embedding.flatten().tolist()
        except Exception as e:
            logger.error(f"Error downloading embedding blob {blob_path}: {e}")
            return None

    def list_blobs(self, prefix: str) -> List[str]:
        """List blob names under a prefix."""
        try:
            blobs = self.container_client.list_blobs(name_starts_with=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Error listing blobs with prefix {prefix}: {e}")
            return []

    def get_summary_data(self, doc_id: str) -> tuple[Optional[str], Optional[List[float]]]:
        """Get summary text and embedding for a document."""
        base_path = f"data/scholarbank/{doc_id}"

        summary_path = f"{base_path}/summary/summary_{doc_id}.txt"
        embedding_path = f"{base_path}/summary_embeddings/summary_embedding_{doc_id}.npz"

        summary_text = self.download_text(summary_path)
        summary_embedding = self.download_embedding(embedding_path)

        return summary_text, summary_embedding

    def get_chunks_with_embeddings(self, doc_id: str) -> List[Dict]:
        """Get chunks that have corresponding embeddings.

        Returns list of dicts with: chunk_id, chunk_text, chunk_embedding
        """
        base_path = f"data/scholarbank/{doc_id}"
        chunks_path = f"{base_path}/chunks/"
        embeddings_path = f"{base_path}/embeddings/"

        # List all chunk files
        chunk_blobs = self.list_blobs(chunks_path)
        embedding_blobs = self.list_blobs(embeddings_path)

        # Extract chunk numbers from filenames
        # Format: chk_<id>_<num>.txt -> embed_<id>_<num>.npz
        chunk_files = {}
        for blob in chunk_blobs:
            if blob.endswith('.txt'):
                # Extract filename like chk_<id>_1.txt
                filename = blob.split('/')[-1]
                if filename.startswith('chk_'):
                    # Get the chunk identifier (e.g., chk_<id>_1)
                    chunk_id = filename.replace('.txt', '')
                    chunk_files[chunk_id] = blob

        embedding_files = {}
        for blob in embedding_blobs:
            if blob.endswith('.npz'):
                filename = blob.split('/')[-1]
                if filename.startswith('embed_'):
                    # embed_<id>_1.npz -> chk_<id>_1
                    embed_id = filename.replace('.npz', '').replace('embed_', 'chk_')
                    embedding_files[embed_id] = blob

        # Match chunks with embeddings
        results = []
        for chunk_id, chunk_path in chunk_files.items():
            if chunk_id in embedding_files:
                chunk_text = self.download_text(chunk_path)
                chunk_embedding = self.download_embedding(embedding_files[chunk_id])

                if chunk_text is not None and chunk_embedding is not None:
                    results.append({
                        'chunk_id': chunk_id,
                        'chunk_text': chunk_text,
                        'chunk_embedding': chunk_embedding
                    })
                else:
                    logger.warning(f"Skipping chunk {chunk_id} due to missing data")
            else:
                logger.debug(f"Chunk {chunk_id} has no corresponding embedding, skipping")

        return results


class SearchClientWrapper:
    """Wrapper for Azure AI Search operations."""

    def __init__(self, config: Dict[str, str]):
        self.endpoint = config['endpoint']
        self.api_key = config['api_key']
        self.credential = AzureKeyCredential(self.api_key)
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=self.credential
        )
        self._search_clients: Dict[str, SearchClient] = {}

    def get_search_client(self, index_name: str) -> SearchClient:
        """Get or create a SearchClient for the given index."""
        if index_name not in self._search_clients:
            self._search_clients[index_name] = SearchClient(
                endpoint=self.endpoint,
                index_name=index_name,
                credential=self.credential
            )
        return self._search_clients[index_name]

    def upload_documents(self, index_name: str, documents: List[Dict]) -> bool:
        """Upload documents to an index. Returns True on success."""
        if not documents:
            return True

        try:
            client = self.get_search_client(index_name)
            result = client.upload_documents(documents=documents)

            # Check if all documents were uploaded successfully
            failed = [r for r in result if not r.succeeded]
            if failed:
                for f in failed:
                    logger.error(f"Failed to upload document {f.key}: {f.error_message}")
                return False

            logger.debug(f"Successfully uploaded {len(documents)} documents to {index_name}")
            return True
        except Exception as e:
            logger.error(f"Error uploading documents to {index_name}: {e}")
            return False

    def delete_documents(self, index_name: str, key_field: str, keys: List[str]) -> bool:
        """Delete documents from an index by their keys."""
        if not keys:
            return True

        try:
            client = self.get_search_client(index_name)
            documents = [{key_field: key} for key in keys]
            result = client.delete_documents(documents=documents)

            failed = [r for r in result if not r.succeeded]
            if failed:
                for f in failed:
                    logger.warning(f"Failed to delete document {f.key}: {f.error_message}")

            logger.debug(f"Deleted {len(keys) - len(failed)} documents from {index_name}")
            return len(failed) == 0
        except Exception as e:
            logger.error(f"Error deleting documents from {index_name}: {e}")
            return False

    def document_exists(self, index_name: str, key_field: str, key_value: str) -> bool:
        """Check if a document exists in an index."""
        try:
            client = self.get_search_client(index_name)
            result = client.search(
                search_text="*",
                filter=f"{key_field} eq '{key_value}'",
                top=1,
                include_total_count=True
            )
            return result.get_count() > 0
        except Exception as e:
            logger.error(f"Error checking document existence in {index_name}: {e}")
            return False

    def get_existing_keys(self, index_name: str, key_field: str, keys: List[str]) -> set:
        """Get set of keys that already exist in the index."""
        if not keys:
            return set()

        existing = set()
        try:
            client = self.get_search_client(index_name)
            # Batch check in chunks to avoid filter length limits
            batch_size = 50
            for i in range(0, len(keys), batch_size):
                batch = keys[i:i+batch_size]
                filter_parts = [f"{key_field} eq '{k}'" for k in batch]
                filter_str = " or ".join(filter_parts)

                result = client.search(
                    search_text="*",
                    filter=filter_str,
                    select=[key_field],
                    top=len(batch)
                )
                for doc in result:
                    existing.add(doc[key_field])
        except Exception as e:
            logger.error(f"Error checking existing keys in {index_name}: {e}")

        return existing
