"""Data processing logic for loading ScholarBank data to AI Search."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from azure_clients import SQLClient, BlobClient, SearchClientWrapper

logger = logging.getLogger(__name__)


@dataclass
class DocumentData:
    """Container for all data related to a document."""
    doc_id: str
    metadata: Dict
    authors: List[Dict]
    departments: List[Dict]
    funders: List[Dict]
    summary_text: Optional[str] = None
    summary_embedding: Optional[List[float]] = None
    chunks: List[Dict] = field(default_factory=list)


@dataclass
class UploadTracker:
    """Track what has been uploaded for rollback purposes."""
    doc_uploaded: bool = False
    chunk_ids_uploaded: List[str] = field(default_factory=list)
    # Authors/departments are globally shared and deduplicated,
    # so we don't roll them back


def get_author_node_id(author_id: Optional[str], doc_id: str, place: int) -> str:
    """Generate author node ID based on the logic.

    If author_id is NULL or empty: 'author_appearance_doc_<doc_id>_<place>'
    Else: 'author_<author_id>'
    """
    if author_id is None or author_id == '':
        return f'author_appearance_doc_{doc_id}_{place}'
    else:
        return f'author_{author_id}'


def extract_year_as_datetime(published_date: Any) -> Optional[str]:
    """Extract year from published_date and format as DateTimeOffset.

    Returns format: 'YYYY-01-01T00:00:00Z'
    """
    if published_date is None:
        return None

    try:
        if isinstance(published_date, datetime):
            year = published_date.year
        elif isinstance(published_date, str):
            # Try to parse various formats
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%Y', '%d-%m-%Y', '%m/%d/%Y']:
                try:
                    dt = datetime.strptime(published_date.strip(), fmt)
                    year = dt.year
                    break
                except ValueError:
                    continue
            else:
                # Try to extract just the year as a number
                year_str = ''.join(c for c in str(published_date) if c.isdigit())[:4]
                if len(year_str) == 4:
                    year = int(year_str)
                else:
                    return None
        elif isinstance(published_date, (int, float)):
            year = int(published_date)
        else:
            return None

        # Validate year is reasonable
        if 1000 <= year <= 9999:
            return f'{year}-01-01T00:00:00Z'
        return None
    except Exception as e:
        logger.warning(f"Could not extract year from published_date: {published_date}, error: {e}")
        return None


class DataLoader:
    """Handles loading data from SQL/Blob to AI Search."""

    def __init__(
        self,
        sql_client: SQLClient,
        blob_client: BlobClient,
        search_client: SearchClientWrapper
    ):
        self.sql_client = sql_client
        self.blob_client = blob_client
        self.search_client = search_client

    def gather_document_data(self, doc_id: str) -> Optional[DocumentData]:
        """Gather all data for a document from SQL and Blob.

        Returns None if critical data is missing.
        """
        # Get metadata from SQL
        metadata = self.sql_client.get_document_metadata(doc_id)
        if not metadata:
            logger.error(f"No metadata found for doc_id: {doc_id}")
            return None

        # Get related data
        authors = self.sql_client.get_authors(doc_id)
        departments = self.sql_client.get_departments(doc_id)
        funders = self.sql_client.get_funders(doc_id)

        # Get summary from Blob
        summary_text, summary_embedding = self.blob_client.get_summary_data(doc_id)
        if summary_text is None or summary_embedding is None:
            logger.error(f"Missing summary data for doc_id: {doc_id}")
            return None

        # Get chunks with embeddings
        chunks = self.blob_client.get_chunks_with_embeddings(doc_id)
        if not chunks:
            logger.warning(f"No chunks with embeddings found for doc_id: {doc_id}")
            # This is not a fatal error, document can still be indexed

        return DocumentData(
            doc_id=doc_id,
            metadata=metadata,
            authors=authors,
            departments=departments,
            funders=funders,
            summary_text=summary_text,
            summary_embedding=summary_embedding,
            chunks=chunks
        )

    def prepare_doc_index_document(self, data: DocumentData) -> Dict:
        """Prepare document for kg-docs-index."""
        doc_id_prefixed = f"doc_{data.doc_id}"

        # Prepare author data
        author_node_ids = []
        author_normalized_names = []
        for author in data.authors:
            node_id = get_author_node_id(
                author.get('author_id'),
                data.doc_id,
                author.get('place', 0)
            )
            author_node_ids.append(node_id)
            if author.get('normalized_name'):
                author_normalized_names.append(author['normalized_name'])

        # Prepare department data
        department_node_ids = []
        department_display_names = []
        for dept in data.departments:
            dept_id = dept.get('department_id')
            if dept_id:
                department_node_ids.append(f"dept_{dept_id}")
            if dept.get('display_name'):
                department_display_names.append(dept['display_name'])

        # Prepare funder data
        funder_normalized_names = []
        funder_display_names = []
        for funder in data.funders:
            if funder.get('normalized_name'):
                funder_normalized_names.append(funder['normalized_name'])
            if funder.get('display_name'):
                funder_display_names.append(funder['display_name'])

        return {
            'doc_id': doc_id_prefixed,
            'node_id': doc_id_prefixed,
            'title': data.metadata.get('title', ''),
            'summary': data.summary_text,
            'summary_vector': data.summary_embedding,
            'url': data.metadata.get('doc_uri', ''),
            'doi': data.metadata.get('doi', ''),
            'doc_type': data.metadata.get('doc_type', ''),
            'language': data.metadata.get('language', ''),
            'publication_year': extract_year_as_datetime(data.metadata.get('published_date')),
            'hidden': False,
            'source': 'sb',
            'author_node_ids': author_node_ids,
            'author_normalized_names': author_normalized_names,
            'department_node_ids': department_node_ids,
            'department_display_names': department_display_names,
            'concept_node_ids': [],
            'concept_labels': [],
            'funder_normalized_names': funder_normalized_names,
            'funder_display_names': funder_display_names
        }

    def prepare_chunk_index_documents(self, data: DocumentData) -> List[Dict]:
        """Prepare documents for kg-chunks-index."""
        doc_id_prefixed = f"doc_{data.doc_id}"

        # Prepare author node IDs
        author_node_ids = []
        for author in data.authors:
            node_id = get_author_node_id(
                author.get('author_id'),
                data.doc_id,
                author.get('place', 0)
            )
            author_node_ids.append(node_id)

        publication_year = extract_year_as_datetime(data.metadata.get('published_date'))

        chunk_docs = []
        for chunk in data.chunks:
            chunk_doc = {
                'chunk_id': chunk['chunk_id'],
                'chunk_text': chunk['chunk_text'],
                'chunk_vector': chunk['chunk_embedding'],
                'doc_id': doc_id_prefixed,
                'doc_node_id': doc_id_prefixed,
                'doc_title': data.metadata.get('title', ''),
                'language': data.metadata.get('language', ''),
                'publication_year': publication_year,
                'hidden': False,
                'source': 'sb',
                'author_node_ids': author_node_ids,
                'concept_node_ids': []
            }
            chunk_docs.append(chunk_doc)

        return chunk_docs

    def prepare_author_index_documents(self, data: DocumentData) -> List[Dict]:
        """Prepare documents for kg-authors-index."""
        author_docs = []
        for author in data.authors:
            node_id = get_author_node_id(
                author.get('author_id'),
                data.doc_id,
                author.get('place', 0)
            )
            author_doc = {
                'author_node_id': node_id,
                'author_id': node_id,
                'normalized_name': author.get('normalized_name', ''),
                'display_name': author.get('display_name', ''),
                'type': author.get('author_type', '')
            }
            author_docs.append(author_doc)

        return author_docs

    def prepare_department_index_documents(self, data: DocumentData) -> List[Dict]:
        """Prepare documents for kg-departments-index."""
        dept_docs = []
        for dept in data.departments:
            dept_id = dept.get('department_id')
            if not dept_id:
                continue

            node_id = f"dept_{dept_id}"
            dept_doc = {
                'department_node_id': node_id,
                'department_id': node_id,
                'normalized_name': dept.get('normalized_name', ''),
                'display_name': dept.get('display_name', '')
            }
            dept_docs.append(dept_doc)

        return dept_docs

    def rollback_uploads(self, tracker: UploadTracker, doc_id_prefixed: str) -> None:
        """Rollback any uploaded data on failure."""
        if tracker.doc_uploaded:
            logger.info(f"Rolling back document: {doc_id_prefixed}")
            self.search_client.delete_documents(
                'kg-docs-index',
                'doc_id',
                [doc_id_prefixed]
            )

        if tracker.chunk_ids_uploaded:
            logger.info(f"Rolling back {len(tracker.chunk_ids_uploaded)} chunks")
            self.search_client.delete_documents(
                'kg-chunks-index',
                'chunk_id',
                tracker.chunk_ids_uploaded
            )

    def upload_document_transactional(self, data: DocumentData) -> bool:
        """Upload a document to all indexes with transactional semantics.

        Returns True if all uploads succeed, False otherwise.
        On failure, any partial uploads are rolled back.
        """
        doc_id_prefixed = f"doc_{data.doc_id}"
        tracker = UploadTracker()

        try:
            # 1. Upload to kg-docs-index
            doc_document = self.prepare_doc_index_document(data)
            if not self.search_client.upload_documents('kg-docs-index', [doc_document]):
                logger.error(f"Failed to upload document to kg-docs-index: {doc_id_prefixed}")
                return False
            tracker.doc_uploaded = True
            logger.debug(f"Uploaded document to kg-docs-index: {doc_id_prefixed}")

            # 2. Upload chunks to kg-chunks-index
            chunk_documents = self.prepare_chunk_index_documents(data)
            if chunk_documents:
                if not self.search_client.upload_documents('kg-chunks-index', chunk_documents):
                    logger.error(f"Failed to upload chunks to kg-chunks-index for: {doc_id_prefixed}")
                    self.rollback_uploads(tracker, doc_id_prefixed)
                    return False
                tracker.chunk_ids_uploaded = [c['chunk_id'] for c in chunk_documents]
                logger.debug(f"Uploaded {len(chunk_documents)} chunks to kg-chunks-index")

            # 3. Upload new authors to kg-authors-index (deduplicated)
            author_documents = self.prepare_author_index_documents(data)
            if author_documents:
                # Check which authors already exist
                author_keys = [a['author_node_id'] for a in author_documents]
                existing_authors = self.search_client.get_existing_keys(
                    'kg-authors-index', 'author_node_id', author_keys
                )
                new_authors = [a for a in author_documents if a['author_node_id'] not in existing_authors]

                if new_authors:
                    if not self.search_client.upload_documents('kg-authors-index', new_authors):
                        logger.error(f"Failed to upload authors to kg-authors-index for: {doc_id_prefixed}")
                        self.rollback_uploads(tracker, doc_id_prefixed)
                        return False
                    logger.debug(f"Uploaded {len(new_authors)} new authors to kg-authors-index")
                else:
                    logger.debug("All authors already exist in index, skipping")

            # 4. Upload new departments to kg-departments-index (deduplicated)
            dept_documents = self.prepare_department_index_documents(data)
            if dept_documents:
                # Check which departments already exist
                dept_keys = [d['department_node_id'] for d in dept_documents]
                existing_depts = self.search_client.get_existing_keys(
                    'kg-departments-index', 'department_node_id', dept_keys
                )
                new_depts = [d for d in dept_documents if d['department_node_id'] not in existing_depts]

                if new_depts:
                    if not self.search_client.upload_documents('kg-departments-index', new_depts):
                        logger.error(f"Failed to upload departments to kg-departments-index for: {doc_id_prefixed}")
                        self.rollback_uploads(tracker, doc_id_prefixed)
                        return False
                    logger.debug(f"Uploaded {len(new_depts)} new departments to kg-departments-index")
                else:
                    logger.debug("All departments already exist in index, skipping")

            return True

        except Exception as e:
            logger.error(f"Exception during upload for {doc_id_prefixed}: {e}")
            self.rollback_uploads(tracker, doc_id_prefixed)
            return False

    def process_document(self, doc_id: str) -> Tuple[bool, str]:
        """Process a single document end-to-end.

        Returns (success: bool, message: str)
        """
        logger.info(f"Processing document: {doc_id}")

        # 1. Gather all data
        data = self.gather_document_data(doc_id)
        if data is None:
            return False, "Failed to gather document data"

        # 2. Upload to all indexes (transactional)
        if not self.upload_document_transactional(data):
            return False, "Failed to upload document to indexes"

        # 3. Update status in SQL
        try:
            self.sql_client.update_ai_search_status(doc_id, True)
        except Exception as e:
            # Rollback the uploads if status update fails
            logger.error(f"Failed to update status, rolling back uploads: {e}")
            doc_id_prefixed = f"doc_{doc_id}"
            tracker = UploadTracker(doc_uploaded=True)
            tracker.chunk_ids_uploaded = [c['chunk_id'] for c in data.chunks]
            self.rollback_uploads(tracker, doc_id_prefixed)
            return False, f"Failed to update status: {e}"

        return True, "Successfully processed"
