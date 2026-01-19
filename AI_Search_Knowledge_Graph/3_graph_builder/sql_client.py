"""Azure SQL operations for graph builder."""
import logging
from typing import List, Dict, Optional
from contextlib import contextmanager
from enum import Enum

import pymssql

from config import config

logger = logging.getLogger(__name__)


class StageStatus(Enum):
    """Stage status values for progress tracking."""
    PENDING = "pending"
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"
    ERRORED = "errored"


class SQLClient:
    """SQL operations for graph building progress and data retrieval."""

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
            tds_version='7.4',
            timeout=30
        )
        try:
            yield conn
        finally:
            conn.close()

    # ========== Document Retrieval ==========

    def get_unique_doc_ids(self) -> List[str]:
        """
        Get unique doc_ids from asm_samples_doc2concepts table.

        Returns:
            List of doc_ids (UUID format, without 'doc_' prefix)
        """
        query = "SELECT DISTINCT doc_id FROM [dbo].[asm_samples_doc2concepts]"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            result = [str(row[0]) for row in cursor.fetchall()]
            logger.info(f"Found {len(result)} unique doc_ids in asm_samples_doc2concepts")
            return result

    def get_doc_concepts(self, doc_id: str) -> List[Dict]:
        """
        Get concept mappings for a document from asm_samples_doc2concepts.

        Args:
            doc_id: Document ID (UUID format, without 'doc_' prefix)

        Returns:
            List of {'concept_id': str, 'strength': float}
        """
        query = """
            SELECT concept_id, strength
            FROM [dbo].[asm_samples_doc2concepts]
            WHERE doc_id = %s
        """
        with self.get_connection() as conn:
            cursor = conn.cursor(as_dict=True)
            cursor.execute(query, (doc_id,))
            return cursor.fetchall()

    # ========== Author Research Interests ==========

    def get_author2concepts(self) -> List[Dict]:
        """
        Get all author-concept mappings from asm_samples_author2concepts.

        Returns:
            List of {'author_id': str, 'concept_id': str, 'num_doc': int}
            - author_id is already in 'author_<uuid>' format
            - concept_id is already prefixed with 'concept_'
        """
        query = """
            SELECT author_id, concept_id, num_doc
            FROM [dbo].[asm_samples_author2concepts]
        """
        with self.get_connection() as conn:
            cursor = conn.cursor(as_dict=True)
            cursor.execute(query)
            result = cursor.fetchall()
            logger.info(f"Found {len(result)} author-concept mappings")
            return result

    # ========== Progress Tracking ==========

    def get_pending_docs(self, max_stage: int = 6) -> List[Dict]:
        """
        Get documents that haven't completed all stages.

        Args:
            max_stage: Maximum stage number (default 6 for stages 0-6)

        Returns:
            List of {'row_id': str, 'current_stage': int, 'stage_status': str, 'retry_count': int}
        """
        query = """
            SELECT row_id,
                   COALESCE(current_stage, 0) as current_stage,
                   COALESCE(stage_status, 'pending') as stage_status,
                   COALESCE(retry_count, 0) as retry_count,
                   err_message
            FROM [dbo].[asm_samples_progress]
            WHERE NOT (COALESCE(current_stage, 0) = %s AND COALESCE(stage_status, 'pending') = 'completed')
              AND COALESCE(retry_count, 0) < 3
            ORDER BY COALESCE(current_stage, 0), row_id
        """
        with self.get_connection() as conn:
            cursor = conn.cursor(as_dict=True)
            cursor.execute(query, (max_stage,))
            return cursor.fetchall()

    def get_all_progress_rows(self) -> List[Dict]:
        """
        Get all progress rows to check initialization status.

        Returns:
            List of {'row_id': str}
        """
        query = "SELECT row_id FROM [dbo].[asm_samples_progress]"
        with self.get_connection() as conn:
            cursor = conn.cursor(as_dict=True)
            cursor.execute(query)
            return cursor.fetchall()

    def update_progress(
        self,
        doc_id: str,
        stage: int,
        status: StageStatus,
        err_message: Optional[str] = None,
        increment_retry: bool = False
    ):
        """
        Update progress for a document.

        Args:
            doc_id: Document ID (row_id in asm_samples_progress)
            stage: Current stage number (0-6)
            status: Stage status
            err_message: Error message if status is ERRORED
            increment_retry: Whether to increment retry_count
        """
        if increment_retry:
            update_sql = """
                UPDATE [dbo].[asm_samples_progress]
                SET current_stage = %s,
                    stage_status = %s,
                    retry_count = COALESCE(retry_count, 0) + 1,
                    err_message = %s
                WHERE row_id = %s
            """
        else:
            update_sql = """
                UPDATE [dbo].[asm_samples_progress]
                SET current_stage = %s,
                    stage_status = %s,
                    err_message = %s
                WHERE row_id = %s
            """

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(update_sql, (stage, status.value, err_message, doc_id))
            conn.commit()

    def mark_stage_in_progress(self, doc_id: str, stage: int):
        """Mark a stage as in-progress."""
        self.update_progress(doc_id, stage, StageStatus.IN_PROGRESS)

    def mark_stage_completed(self, doc_id: str, stage: int):
        """Mark a stage as completed."""
        self.update_progress(doc_id, stage, StageStatus.COMPLETED)

    def mark_stage_errored(self, doc_id: str, stage: int, err_message: str):
        """Mark a stage as errored and increment retry count."""
        self.update_progress(
            doc_id, stage, StageStatus.ERRORED,
            err_message=err_message[:500] if err_message else None,  # Truncate to fit column
            increment_retry=True
        )

    def get_doc_progress(self, doc_id: str) -> Optional[Dict]:
        """
        Get progress for a specific document.

        Args:
            doc_id: Document ID (row_id)

        Returns:
            Dict with current_stage, stage_status, retry_count, err_message
            or None if not found
        """
        query = """
            SELECT row_id,
                   COALESCE(current_stage, 0) as current_stage,
                   COALESCE(stage_status, 'pending') as stage_status,
                   COALESCE(retry_count, 0) as retry_count,
                   err_message
            FROM [dbo].[asm_samples_progress]
            WHERE row_id = %s
        """
        with self.get_connection() as conn:
            cursor = conn.cursor(as_dict=True)
            cursor.execute(query, (doc_id,))
            result = cursor.fetchone()
            return result
