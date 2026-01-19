"""Azure SQL operations for sample-to-concept mapping."""
import logging
from typing import List, Dict, Set
from contextlib import contextmanager

import pymssql

from config import config

logger = logging.getLogger(__name__)


class SQLClient:
    """SQL operations for document-concept and author-concept mappings."""

    def __init__(self):
        self.server = config.sql_server
        self.database = config.sql_database
        self.username = config.sql_username
        self.password = config.sql_password

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = pymssql.connect(
            server=self.server,
            user=self.username,
            password=self.password,
            database=self.database,
            timeout=30
        )
        try:
            yield conn
        finally:
            conn.close()

    def get_eligible_documents(self) -> List[str]:
        """
        Get document row_ids ready for concept mapping.
        Query: current_stage=4, stage_status='completed', is_validated=1, in_ai_search=1
        """
        query = """
            SELECT row_id
            FROM [dbo].[sb_processed_status]
            WHERE current_stage = 4
              AND stage_status = 'completed'
              AND is_validated = 1
              AND in_ai_search = 1
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            results = [str(row[0]) for row in cursor.fetchall()]
            logger.info(f"Found {len(results)} eligible documents")
            return results

    def get_processed_doc_ids(self) -> Set[str]:
        """
        Get doc_ids already in asm_samples_doc2concepts for resumability.
        """
        query = "SELECT DISTINCT doc_id FROM [dbo].[asm_samples_doc2concepts]"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            result = {str(row[0]) for row in cursor.fetchall()}
            logger.info(f"Found {len(result)} already processed documents")
            return result

    def get_all_processed_doc_ids_list(self) -> List[str]:
        """
        Get all doc_ids from asm_samples_doc2concepts as a list.
        Used for backfill operations.
        """
        query = "SELECT DISTINCT doc_id FROM [dbo].[asm_samples_doc2concepts]"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            result = [str(row[0]) for row in cursor.fetchall()]
            logger.info(f"Found {len(result)} documents for backfill")
            return result

    def get_doc_concepts(self, doc_id: str) -> List[str]:
        """
        Get concept_ids for a document from asm_samples_doc2concepts.
        Used for backfill operations.
        """
        query = "SELECT concept_id FROM [dbo].[asm_samples_doc2concepts] WHERE doc_id = %s"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (doc_id,))
            return [str(row[0]) for row in cursor.fetchall()]

    def upsert_author2concept(self, author_id: str, concept_id: str) -> None:
        """
        Upsert a single author-concept mapping.
        If exists, increment num_doc. Otherwise insert with num_doc=1.
        """
        merge_sql = """
            MERGE [dbo].[asm_samples_author2concepts] AS target
            USING (SELECT %s AS author_id, %s AS concept_id) AS source
            ON (target.author_id = source.author_id AND target.concept_id = source.concept_id)
            WHEN MATCHED THEN
                UPDATE SET num_doc = target.num_doc + 1
            WHEN NOT MATCHED THEN
                INSERT (author_id, concept_id, num_doc)
                VALUES (source.author_id, source.concept_id, 1);
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(merge_sql, (author_id, concept_id))
            conn.commit()

    def _delete_doc_entries(self, doc_id: str, cursor) -> None:
        """
        Delete all existing entries for a doc_id (for re-processing).
        Called within a transaction.
        """
        cursor.execute(
            "DELETE FROM [dbo].[asm_samples_doc2concepts] WHERE doc_id = %s",
            (doc_id,)
        )
        deleted_count = cursor.rowcount
        if deleted_count > 0:
            logger.debug(f"Deleted {deleted_count} existing doc2concepts entries for {doc_id}")

    def _insert_doc2concepts(
        self,
        doc_id: str,
        concepts: List[Dict],
        cursor
    ) -> None:
        """
        Insert document-to-concept mappings.
        Schema: (doc_id, concept_id, strength) with PK (doc_id, concept_id)
        """
        insert_sql = """
            INSERT INTO [dbo].[asm_samples_doc2concepts]
            (doc_id, concept_id, strength)
            VALUES (%s, %s, %s)
        """
        for concept in concepts:
            cursor.execute(insert_sql, (
                doc_id,
                concept['concept_id'],
                concept['strength']
            ))
        logger.debug(f"Inserted {len(concepts)} doc2concepts for {doc_id}")

    def _upsert_author2concepts(
        self,
        author_ids: List[str],
        concepts: List[Dict],
        cursor
    ) -> None:
        """
        Update/insert author-to-concept mappings.
        Schema: (author_id, concept_id, num_doc) with PK (author_id, concept_id)

        Logic: If entry exists, increment num_doc; else insert with num_doc=1
        """
        merge_sql = """
            MERGE [dbo].[asm_samples_author2concepts] AS target
            USING (SELECT %s AS author_id, %s AS concept_id) AS source
            ON (target.author_id = source.author_id AND target.concept_id = source.concept_id)
            WHEN MATCHED THEN
                UPDATE SET num_doc = target.num_doc + 1
            WHEN NOT MATCHED THEN
                INSERT (author_id, concept_id, num_doc)
                VALUES (source.author_id, source.concept_id, 1);
        """
        upsert_count = 0
        for author_id in author_ids:
            for concept in concepts:
                cursor.execute(merge_sql, (author_id, concept['concept_id']))
                upsert_count += 1

        if author_ids:
            logger.debug(f"Upserted {upsert_count} author2concepts for {len(author_ids)} authors")

    def process_document_transaction(
        self,
        doc_id: str,
        author_ids: List[str],
        concepts: List[Dict]
    ) -> bool:
        """
        All-or-nothing transaction for a document.
        1. Delete existing doc entries
        2. Insert new doc2concepts
        3. Upsert author2concepts

        Args:
            doc_id: Document ID
            author_ids: List of author node IDs (filtered to author_xxxx format)
            concepts: List of {concept_id, strength}

        Returns:
            True on success

        Raises:
            Exception on failure (transaction rolled back)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                # Delete old entries for this doc
                self._delete_doc_entries(doc_id, cursor)

                # Insert new doc2concepts
                self._insert_doc2concepts(doc_id, concepts, cursor)

                # Upsert author2concepts
                self._upsert_author2concepts(author_ids, concepts, cursor)

                conn.commit()
                logger.debug(f"Transaction committed for {doc_id}")
                return True

            except Exception as e:
                conn.rollback()
                logger.error(f"Transaction failed for {doc_id}: {e}")
                raise
