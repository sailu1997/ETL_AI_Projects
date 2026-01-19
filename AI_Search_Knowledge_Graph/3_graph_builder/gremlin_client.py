"""Gremlin client for CosmosDB graph operations."""
import logging
import re
from typing import List, Optional, Any

from gremlin_python.driver import client, serializer

from config import config

logger = logging.getLogger(__name__)

# Regex pattern for author_<uuid> format
# Matches: author_97bc05d3-48cc-4cbf-afd1-1313e3e552e0
# Does not match: author_appearance_doc_*
AUTHOR_UUID_PATTERN = re.compile(r'^author_[a-f0-9-]{36}$')


class GremlinClient:
    """Client for CosmosDB Gremlin operations."""

    def __init__(self):
        self._client = None
        self._connect()

    def _connect(self):
        """Establish connection to CosmosDB Gremlin."""
        self._client = client.Client(
            url=config.gremlin_endpoint,
            traversal_source='g',
            username=f"/dbs/{config.gremlin_database}/colls/{config.gremlin_collection}",
            password=config.gremlin_key,
            message_serializer=serializer.GraphSONSerializersV2d0()
        )
        logger.info(f"Gremlin client connected to {config.gremlin_collection}")

    def close(self):
        """Close the Gremlin connection."""
        if self._client:
            self._client.close()
            logger.info("Gremlin client closed")

    def _escape(self, text: str) -> str:
        """Escape special characters for Gremlin queries."""
        if text is None:
            return ""
        text = str(text)
        return (text
                .replace("\\", "\\\\")
                .replace("'", "\\'")
                .replace('"', '\\"')
                .replace('\n', '\\n')
                .replace('\r', '\\r'))

    def execute(self, query: str) -> List[Any]:
        """Execute a Gremlin query and return results."""
        try:
            result = self._client.submit(query).all().result()
            return result if result else []
        except Exception as e:
            logger.error(f"Gremlin query failed: {e}")
            logger.debug(f"Failed query: {query}")
            raise

    # ========== Node Existence Checks ==========

    def node_exists(self, node_id: str) -> bool:
        """Check if a node exists by ID."""
        query = f"g.V('{self._escape(node_id)}').count()"
        result = self.execute(query)
        return result and result[0] > 0

    def edge_exists(self, from_id: str, to_id: str, label: str) -> bool:
        """Check if an edge exists between two nodes."""
        query = f"""g.V('{self._escape(from_id)}').outE('{label}').where(inV().hasId('{self._escape(to_id)}')).count()"""
        result = self.execute(query)
        return result and result[0] > 0

    # ========== Node Creation ==========

    def create_doc_node(self, node_id: str, title: str, source: str, hidden: bool) -> bool:
        """
        Create a document node if it doesn't exist.

        Args:
            node_id: Document node ID (format: doc_<uuid>)
            title: Document title
            source: Source identifier ('sb' or 'dg')
            hidden: Whether the document should be hidden

        Returns:
            True if created, False if already exists
        """
        if self.node_exists(node_id):
            logger.debug(f"Doc node already exists: {node_id}")
            return False

        hidden_str = 'true' if hidden else 'false'
        query = f"""g.addV('doc').property('id', '{self._escape(node_id)}').property('pk', '{self._escape(node_id)}').property('node_id', '{self._escape(node_id)}').property('title', '{self._escape(title)}').property('source', '{self._escape(source)}').property('hidden', {hidden_str})"""
        self.execute(query)
        logger.debug(f"Created doc node: {node_id}")
        return True

    def create_author_node(
        self,
        node_id: str,
        author_id: Optional[str],
        normalized_name: str,
        display_name: str,
        author_type: str,
        place: Optional[int]
    ) -> bool:
        """
        Create an author node if it doesn't exist.

        Args:
            node_id: Author node ID (format: author_<uuid> or author_appearance_<...>)
            author_id: Stable author ID if available
            normalized_name: Canonical name for search
            display_name: Full formatted name
            author_type: 'author' or 'supervisor'
            place: Author order on paper (1-based)

        Returns:
            True if created, False if already exists
        """
        if self.node_exists(node_id):
            logger.debug(f"Author node already exists: {node_id}")
            return False

        # Build property chain
        props = [
            f".property('id', '{self._escape(node_id)}')",
            f".property('pk', '{self._escape(node_id)}')",
            f".property('node_id', '{self._escape(node_id)}')",
            f".property('normalized_name', '{self._escape(normalized_name)}')",
            f".property('display_name', '{self._escape(display_name)}')",
            f".property('type', '{self._escape(author_type)}')"
        ]

        if author_id:
            props.append(f".property('author_id', '{self._escape(author_id)}')")

        if place is not None:
            props.append(f".property('place', {place})")

        query = f"g.addV('author'){''.join(props)}"
        self.execute(query)
        logger.debug(f"Created author node: {node_id}")
        return True

    def create_department_node(
        self,
        node_id: str,
        department_id: str,
        normalized_name: str,
        display_name: str
    ) -> bool:
        """
        Create a department node if it doesn't exist.

        Args:
            node_id: Department node ID (format: department_<id>)
            department_id: Internal department ID
            normalized_name: Canonical name
            display_name: User-facing label

        Returns:
            True if created, False if already exists
        """
        if self.node_exists(node_id):
            logger.debug(f"Department node already exists: {node_id}")
            return False

        query = f"""g.addV('department').property('id', '{self._escape(node_id)}').property('pk', '{self._escape(node_id)}').property('node_id', '{self._escape(node_id)}').property('department_id', '{self._escape(department_id)}').property('normalized_name', '{self._escape(normalized_name)}').property('display_name', '{self._escape(display_name)}')"""
        self.execute(query)
        logger.debug(f"Created department node: {node_id}")
        return True

    # ========== Edge Creation ==========

    def create_tagged_as_edge(self, doc_id: str, concept_id: str, strength: float) -> bool:
        """
        Create TAGGED_AS edge (doc -> concept) with strength property.

        Args:
            doc_id: Document node ID
            concept_id: Concept node ID
            strength: Tag strength (0.0-10.0)

        Returns:
            True if created, False if already exists
        """
        if self.edge_exists(doc_id, concept_id, 'TAGGED_AS'):
            logger.debug(f"TAGGED_AS edge already exists: {doc_id} -> {concept_id}")
            return False

        query = f"""g.V('{self._escape(doc_id)}').addE('TAGGED_AS').to(g.V('{self._escape(concept_id)}')).property('strength', {strength})"""
        self.execute(query)
        logger.debug(f"Created TAGGED_AS edge: {doc_id} -> {concept_id} (strength={strength})")
        return True

    def create_writes_edge(self, author_id: str, doc_id: str) -> bool:
        """
        Create WRITES edge (author -> doc).

        Args:
            author_id: Author node ID
            doc_id: Document node ID

        Returns:
            True if created, False if already exists
        """
        if self.edge_exists(author_id, doc_id, 'WRITES'):
            logger.debug(f"WRITES edge already exists: {author_id} -> {doc_id}")
            return False

        query = f"""g.V('{self._escape(author_id)}').addE('WRITES').to(g.V('{self._escape(doc_id)}'))"""
        self.execute(query)
        logger.debug(f"Created WRITES edge: {author_id} -> {doc_id}")
        return True

    def create_coauthors_with_edge(self, author1_id: str, author2_id: str) -> bool:
        """
        Create COAUTHORS_WITH edge (author -> author).

        Only creates edge between authors with author_<uuid> format.
        Uses alphabetical ordering (lower ID -> higher ID) to ensure
        single direction and prevent duplicate edges.

        Args:
            author1_id: First author node ID
            author2_id: Second author node ID

        Returns:
            True if created, False if skipped or already exists
        """
        # Validate both are in author_<uuid> format
        if not AUTHOR_UUID_PATTERN.match(author1_id) or not AUTHOR_UUID_PATTERN.match(author2_id):
            logger.debug(f"Skipping COAUTHORS_WITH: not both author_<uuid> format")
            return False

        # Same author - skip
        if author1_id == author2_id:
            return False

        # Ensure consistent direction (lower ID -> higher ID)
        if author1_id > author2_id:
            author1_id, author2_id = author2_id, author1_id

        if self.edge_exists(author1_id, author2_id, 'COAUTHORS_WITH'):
            logger.debug(f"COAUTHORS_WITH edge already exists: {author1_id} -> {author2_id}")
            return False

        query = f"""g.V('{self._escape(author1_id)}').addE('COAUTHORS_WITH').to(g.V('{self._escape(author2_id)}'))"""
        self.execute(query)
        logger.debug(f"Created COAUTHORS_WITH edge: {author1_id} -> {author2_id}")
        return True

    def create_produces_edge(self, department_id: str, doc_id: str) -> bool:
        """
        Create PRODUCES edge (department -> doc).

        Args:
            department_id: Department node ID
            doc_id: Document node ID

        Returns:
            True if created, False if already exists
        """
        if self.edge_exists(department_id, doc_id, 'PRODUCES'):
            logger.debug(f"PRODUCES edge already exists: {department_id} -> {doc_id}")
            return False

        query = f"""g.V('{self._escape(department_id)}').addE('PRODUCES').to(g.V('{self._escape(doc_id)}'))"""
        self.execute(query)
        logger.debug(f"Created PRODUCES edge: {department_id} -> {doc_id}")
        return True

    def create_has_research_interest_edge(self, author_id: str, concept_id: str, count: int) -> bool:
        """
        Create HAS_RESEARCH_INTEREST edge (author -> concept) with count property.

        Args:
            author_id: Author node ID
            concept_id: Concept node ID
            count: Number of documents this author has written about this concept

        Returns:
            True if created, False if already exists
        """
        if self.edge_exists(author_id, concept_id, 'HAS_RESEARCH_INTEREST'):
            logger.debug(f"HAS_RESEARCH_INTEREST edge already exists: {author_id} -> {concept_id}")
            return False

        query = f"""g.V('{self._escape(author_id)}').addE('HAS_RESEARCH_INTEREST').to(g.V('{self._escape(concept_id)}')).property('count', {count})"""
        self.execute(query)
        logger.debug(f"Created HAS_RESEARCH_INTEREST edge: {author_id} -> {concept_id} (count={count})")
        return True
