"""Author research interests graph building (Part 2)."""
import logging
from typing import Tuple

from gremlin_client import GremlinClient, AUTHOR_UUID_PATTERN
from search_client import SearchClient
from sql_client import SQLClient

logger = logging.getLogger(__name__)


class AuthorInterestBuilder:
    """
    Builds HAS_RESEARCH_INTEREST edges from asm_samples_author2concepts table.

    - author_id is already in 'author_<uuid>' format
    - concept_id is already prefixed with 'concept_'
    - Creates author node if it doesn't exist
    - Idempotent: checks edge existence before creating
    """

    def __init__(self):
        self.gremlin = GremlinClient()
        self.search = SearchClient()
        self.sql = SQLClient()

    def close(self):
        """Clean up resources."""
        self.gremlin.close()

    def build_research_interests(self) -> Tuple[int, int, int]:
        """
        Process all author-concept mappings from asm_samples_author2concepts.

        Returns:
            Tuple of (total_processed, edges_created, errors)
        """
        mappings = self.sql.get_author2concepts()
        total = len(mappings)
        created = 0
        skipped = 0
        errors = 0

        logger.info(f"Processing {total} author-concept mappings")

        for idx, mapping in enumerate(mappings, 1):
            author_id = mapping['author_id']
            concept_id = mapping['concept_id']
            num_doc = mapping['num_doc']

            try:
                # Check if concept node exists
                if not self.gremlin.node_exists(concept_id):
                    logger.debug(f"Concept node does not exist: {concept_id}, skipping")
                    skipped += 1
                    continue

                # Ensure author node exists
                if not self.gremlin.node_exists(author_id):
                    # Try to get author info from AI Search
                    author_data = self.search.get_author(author_id)

                    if author_data:
                        self.gremlin.create_author_node(
                            node_id=author_id,
                            author_id=author_data.get('author_id'),
                            normalized_name=author_data.get('normalized_name', ''),
                            display_name=author_data.get('display_name', ''),
                            author_type=author_data.get('type', 'author'),
                            place=None
                        )
                    else:
                        # Create minimal author node
                        # Extract UUID from author_<uuid> format
                        if AUTHOR_UUID_PATTERN.match(author_id):
                            raw_author_id = author_id.replace('author_', '')
                        else:
                            raw_author_id = None

                        self.gremlin.create_author_node(
                            node_id=author_id,
                            author_id=raw_author_id,
                            normalized_name=author_id,
                            display_name=author_id,
                            author_type='author',
                            place=None
                        )
                    logger.debug(f"Created author node: {author_id}")

                # Create HAS_RESEARCH_INTEREST edge
                if self.gremlin.create_has_research_interest_edge(author_id, concept_id, num_doc):
                    created += 1

                if idx % 100 == 0:
                    logger.info(f"Progress: {idx}/{total} processed, {created} edges created, {skipped} skipped")

            except Exception as e:
                logger.error(f"Failed to process {author_id} -> {concept_id}: {e}")
                errors += 1

        logger.info(f"Completed: {total} processed, {created} edges created, {skipped} skipped, {errors} errors")
        return total, created, errors
