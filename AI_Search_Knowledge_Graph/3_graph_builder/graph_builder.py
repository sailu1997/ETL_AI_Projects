"""Core graph building logic for document processing stages 0-6."""
import logging
from typing import List, Optional, Tuple
from itertools import combinations

from gremlin_client import GremlinClient, AUTHOR_UUID_PATTERN
from search_client import SearchClient
from sql_client import SQLClient

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds knowledge graph from documents.

    Stages:
    0: Create doc node
    1: Create TAGGED_AS edges (doc -> concept)
    2: Create author nodes
    3: Create WRITES edges (author -> doc)
    4: Create COAUTHORS_WITH edges (between author_<uuid> authors)
    5: Create department nodes
    6: Create PRODUCES edges (department -> doc)
    """

    def __init__(self):
        self.gremlin = GremlinClient()
        self.search = SearchClient()
        self.sql = SQLClient()

    def close(self):
        """Clean up resources."""
        self.gremlin.close()

    def process_document(self, doc_id: str, start_stage: int = 0) -> Tuple[bool, int, Optional[str]]:
        """
        Process a document through all stages from start_stage.

        Args:
            doc_id: Document ID (UUID format, without 'doc_' prefix)
            start_stage: Stage to start from (for resumption)

        Returns:
            Tuple of (success, last_completed_stage, error_message)
        """
        node_id = f"doc_{doc_id}"

        # Stage 0: Create doc node
        if start_stage <= 0:
            success, err = self._stage_0_create_doc_node(doc_id, node_id)
            if not success:
                return False, 0, err

        # Stage 1: Create TAGGED_AS edges
        if start_stage <= 1:
            success, err = self._stage_1_create_tagged_as_edges(doc_id, node_id)
            if not success:
                return False, 1, err

        # Fetch author and department info once for stages 2-6
        doc_data = self.search.get_document(node_id)
        if not doc_data:
            return False, 1, f"Document not found in AI Search: {node_id}"

        author_node_ids = doc_data.get('author_node_ids') or []
        department_node_ids = doc_data.get('department_node_ids') or []

        # Stage 2: Create author nodes
        if start_stage <= 2:
            success, err = self._stage_2_create_author_nodes(author_node_ids)
            if not success:
                return False, 2, err

        # Stage 3: Create WRITES edges
        if start_stage <= 3:
            success, err = self._stage_3_create_writes_edges(author_node_ids, node_id)
            if not success:
                return False, 3, err

        # Stage 4: Create COAUTHORS_WITH edges
        if start_stage <= 4:
            success, err = self._stage_4_create_coauthors_edges(author_node_ids)
            if not success:
                return False, 4, err

        # Stage 5: Create department nodes
        if start_stage <= 5:
            success, err = self._stage_5_create_department_nodes(department_node_ids)
            if not success:
                return False, 5, err

        # Stage 6: Create PRODUCES edges
        if start_stage <= 6:
            success, err = self._stage_6_create_produces_edges(department_node_ids, node_id)
            if not success:
                return False, 6, err

        return True, 6, None

    def _stage_0_create_doc_node(self, doc_id: str, node_id: str) -> Tuple[bool, Optional[str]]:
        """Stage 0: Create document node."""
        try:
            # Look up doc in AI Search
            doc_data = self.search.get_document(node_id)
            if not doc_data:
                return False, f"Document not found in AI Search: {node_id}"

            title = doc_data.get('title', '')
            source = doc_data.get('source', 'sb')
            hidden = doc_data.get('hidden', False) or False

            self.gremlin.create_doc_node(node_id, title, source, hidden)
            logger.info(f"Stage 0 completed: doc node {node_id}")
            return True, None
        except Exception as e:
            logger.error(f"Stage 0 failed for {node_id}: {e}")
            return False, str(e)

    def _stage_1_create_tagged_as_edges(self, doc_id: str, node_id: str) -> Tuple[bool, Optional[str]]:
        """Stage 1: Create TAGGED_AS edges from doc to concepts."""
        try:
            concepts = self.sql.get_doc_concepts(doc_id)
            created = 0
            skipped = 0

            for c in concepts:
                concept_id = c['concept_id']
                strength = float(c['strength'])

                # Check if concept node exists
                if not self.gremlin.node_exists(concept_id):
                    logger.warning(f"Concept node does not exist: {concept_id}, skipping TAGGED_AS edge")
                    skipped += 1
                    continue

                if self.gremlin.create_tagged_as_edge(node_id, concept_id, strength):
                    created += 1

            logger.info(f"Stage 1 completed: {created} TAGGED_AS edges created, {skipped} skipped for {node_id}")
            return True, None
        except Exception as e:
            logger.error(f"Stage 1 failed for {node_id}: {e}")
            return False, str(e)

    def _stage_2_create_author_nodes(self, author_node_ids: List[str]) -> Tuple[bool, Optional[str]]:
        """Stage 2: Create author nodes."""
        try:
            created = 0
            for author_node_id in author_node_ids:
                # Fetch author details from AI Search
                author_data = self.search.get_author(author_node_id)

                if author_data:
                    # Use data from AI Search
                    author_id = author_data.get('author_id')
                    normalized_name = author_data.get('normalized_name', '')
                    display_name = author_data.get('display_name', '')
                    author_type = author_data.get('type', 'author')
                else:
                    # Author not in index - create with minimal info
                    logger.debug(f"Author not in AI Search index: {author_node_id}")
                    # Parse author_id from node_id if possible
                    if AUTHOR_UUID_PATTERN.match(author_node_id):
                        author_id = author_node_id.replace('author_', '')
                    else:
                        author_id = None
                    normalized_name = author_node_id
                    display_name = author_node_id
                    author_type = 'author'

                if self.gremlin.create_author_node(
                    node_id=author_node_id,
                    author_id=author_id,
                    normalized_name=normalized_name,
                    display_name=display_name,
                    author_type=author_type,
                    place=None  # Place not available from AI Search index
                ):
                    created += 1

            logger.info(f"Stage 2 completed: {created} author nodes created")
            return True, None
        except Exception as e:
            logger.error(f"Stage 2 failed: {e}")
            return False, str(e)

    def _stage_3_create_writes_edges(self, author_node_ids: List[str], doc_node_id: str) -> Tuple[bool, Optional[str]]:
        """Stage 3: Create WRITES edges from authors to doc."""
        try:
            created = 0
            for author_node_id in author_node_ids:
                if self.gremlin.create_writes_edge(author_node_id, doc_node_id):
                    created += 1

            logger.info(f"Stage 3 completed: {created} WRITES edges for {doc_node_id}")
            return True, None
        except Exception as e:
            logger.error(f"Stage 3 failed for {doc_node_id}: {e}")
            return False, str(e)

    def _stage_4_create_coauthors_edges(self, author_node_ids: List[str]) -> Tuple[bool, Optional[str]]:
        """Stage 4: Create COAUTHORS_WITH edges between author_<uuid> authors."""
        try:
            # Filter to only author_<uuid> format
            uuid_authors = [a for a in author_node_ids if AUTHOR_UUID_PATTERN.match(a)]

            if len(uuid_authors) < 2:
                logger.info(f"Stage 4 completed: 0 COAUTHORS_WITH edges (< 2 author_uuid authors)")
                return True, None

            created = 0
            # Generate all pairs (combinations ensures no duplicates)
            for author1, author2 in combinations(uuid_authors, 2):
                if self.gremlin.create_coauthors_with_edge(author1, author2):
                    created += 1

            logger.info(f"Stage 4 completed: {created} COAUTHORS_WITH edges")
            return True, None
        except Exception as e:
            logger.error(f"Stage 4 failed: {e}")
            return False, str(e)

    def _stage_5_create_department_nodes(self, department_node_ids: List[str]) -> Tuple[bool, Optional[str]]:
        """Stage 5: Create department nodes."""
        try:
            created = 0
            for dept_node_id in department_node_ids:
                # Fetch department details from AI Search
                dept_data = self.search.get_department(dept_node_id)

                if dept_data:
                    department_id = dept_data.get('department_id', '')
                    normalized_name = dept_data.get('normalized_name', '')
                    display_name = dept_data.get('display_name', '')
                else:
                    # Department not in index - create with minimal info
                    logger.debug(f"Department not in AI Search index: {dept_node_id}")
                    department_id = dept_node_id.replace('department_', '')
                    normalized_name = dept_node_id
                    display_name = dept_node_id

                if self.gremlin.create_department_node(
                    node_id=dept_node_id,
                    department_id=department_id,
                    normalized_name=normalized_name,
                    display_name=display_name
                ):
                    created += 1

            logger.info(f"Stage 5 completed: {created} department nodes")
            return True, None
        except Exception as e:
            logger.error(f"Stage 5 failed: {e}")
            return False, str(e)

    def _stage_6_create_produces_edges(self, department_node_ids: List[str], doc_node_id: str) -> Tuple[bool, Optional[str]]:
        """Stage 6: Create PRODUCES edges from departments to doc."""
        try:
            created = 0
            for dept_node_id in department_node_ids:
                if self.gremlin.create_produces_edge(dept_node_id, doc_node_id):
                    created += 1

            logger.info(f"Stage 6 completed: {created} PRODUCES edges for {doc_node_id}")
            return True, None
        except Exception as e:
            logger.error(f"Stage 6 failed for {doc_node_id}: {e}")
            return False, str(e)
