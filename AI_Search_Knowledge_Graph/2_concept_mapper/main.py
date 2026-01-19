#!/usr/bin/env python3
"""Main entry point for ScholarBank sample-to-concept mapping.

Usage:
    python main.py [--limit N] [--debug] [--doc-id DOC_ID]
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from typing import List, Dict, Set

from config import config
from sql_client import SQLClient
from search_client import SearchClient
from llm_judge import LLMJudge

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Orchestrates document-to-concept mapping."""

    def __init__(self):
        self.sql_client = SQLClient()
        self.search_client = SearchClient()
        self.llm_judge = LLMJudge()

        self.processed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.failed_docs: List[Dict] = []

    def get_docs_to_process(self) -> List[str]:
        """
        Get eligible documents minus already processed ones.
        Implements resumability.
        """
        # Get all eligible documents
        all_doc_ids = self.sql_client.get_eligible_documents()
        logger.info(f"Found {len(all_doc_ids)} eligible documents from sb_processed_status")

        # Get already processed doc_ids
        processed = self.sql_client.get_processed_doc_ids()
        self.skipped_count = len(processed)

        # Filter to unprocessed
        to_process = [d for d in all_doc_ids if d not in processed]
        logger.info(f"Documents to process: {len(to_process)} (skipping {len(processed)} already done)")

        return to_process

    def process_document(self, doc_id: str) -> bool:
        """
        Process a single document end-to-end.

        Steps:
        1. Retrieve doc from kg-docs-index
        2. Filter author_node_ids
        3. Hybrid search for concepts
        4. LLM judge evaluation
        5. Store results (transaction)

        Returns:
            True on success, False on failure
        """
        start_time = time.time()
        logger.info(f"Processing document: {doc_id}")

        try:
            # Step 1: Get document from AI Search
            doc = self.search_client.get_document(doc_id)
            if not doc:
                raise ValueError(f"Document not found in AI Search: {doc_id}")

            title = doc.get('title', '')
            summary = doc.get('summary', '')
            summary_vector = doc.get('summary_vector', [])
            author_node_ids = doc.get('author_node_ids', [])

            if not summary_vector:
                raise ValueError(f"Document missing summary_vector: {doc_id}")

            logger.debug(f"Document: title='{title[:50]}...', summary_len={len(summary)}")

            # Step 2: Filter authors (only 'author_xxxx', not 'author_appearance_xxxx')
            filtered_authors = self.search_client.filter_author_node_ids(author_node_ids or [])
            logger.debug(f"Filtered authors: {len(filtered_authors)}")

            # Step 3: Hybrid search for concepts
            candidates = self.search_client.search_concepts_hybrid(
                query_text=summary[:1000] if summary else title,
                embedding=summary_vector,
                top_k=config.hybrid_search_top_k  # 50
            )

            if not candidates:
                raise ValueError(f"No concept candidates found for: {doc_id}")

            logger.info(f"Retrieved {len(candidates)} concept candidates")

            # Step 4: LLM judge evaluation
            evaluated, metadata = self.llm_judge.judge_concepts(
                doc_title=title,
                doc_summary=summary,
                candidate_concepts=candidates,
                top_k=config.llm_judge_top_k  # 20
            )

            # Keep only top 15
            final_concepts = evaluated[:config.final_top_k]  # 15

            if not final_concepts:
                raise ValueError(f"LLM judge returned no valid concepts for: {doc_id}")

            logger.info(
                f"LLM judge: {len(final_concepts)} concepts "
                f"(attempts={metadata.get('attempts_made')}, "
                f"consolidation={metadata.get('consolidation_used')})"
            )

            # Prepare data for storage
            # strength = LLM score (0-10 scale as specified)
            concepts_for_storage = [
                {
                    'concept_id': c['concept_id'],
                    'strength': c['score']  # Keep 0-10 scale
                }
                for c in final_concepts
            ]

            # Step 5: Store in SQL (all-or-nothing transaction)
            self.sql_client.process_document_transaction(
                doc_id=doc_id,
                author_ids=filtered_authors,
                concepts=concepts_for_storage
            )

            elapsed = time.time() - start_time
            logger.info(f"SUCCESS: {doc_id} processed in {elapsed:.2f}s")
            self.processed_count += 1
            return True

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"FAILED: {doc_id} after {elapsed:.2f}s - {e}")
            self.failed_count += 1
            self.failed_docs.append({
                'doc_id': doc_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False

    def run(self, limit: int = None) -> Dict:
        """
        Main processing loop.

        Args:
            limit: Optional limit on documents to process

        Returns:
            Summary statistics
        """
        start_time = time.time()

        # Get documents to process
        doc_ids = self.get_docs_to_process()

        if limit:
            doc_ids = doc_ids[:limit]
            logger.info(f"Limited to {limit} documents")

        total = len(doc_ids)

        if total == 0:
            logger.info("No documents to process - all eligible docs already completed")
            return {
                'total_eligible': self.skipped_count,
                'to_process': 0,
                'processed': 0,
                'failed': 0,
                'skipped': self.skipped_count
            }

        logger.info(f"Starting processing of {total} documents")
        logger.info("=" * 60)

        # Process each document
        for idx, doc_id in enumerate(doc_ids, 1):
            logger.info(f"[{idx}/{total}] Processing {doc_id}")

            try:
                self.process_document(doc_id)
            except Exception as e:
                # Step-over on failure
                logger.error(f"Unhandled exception for {doc_id}: {e}")
                if doc_id not in [d['doc_id'] for d in self.failed_docs]:
                    self.failed_count += 1
                    self.failed_docs.append({
                        'doc_id': doc_id,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })

        # Save failed docs
        if self.failed_docs:
            self._save_failed_docs()

        # Generate summary
        elapsed = time.time() - start_time
        summary = {
            'total_eligible': total + self.skipped_count,
            'to_process': total,
            'processed': self.processed_count,
            'failed': self.failed_count,
            'skipped': self.skipped_count,
            'duration_seconds': round(elapsed, 2),
            'success_rate': round((self.processed_count / total * 100), 1) if total > 0 else 0
        }

        self._print_summary(summary)
        return summary

    def _save_failed_docs(self):
        """Save failed documents to JSON file."""
        with open(config.failed_docs_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'count': len(self.failed_docs),
                'documents': self.failed_docs
            }, f, indent=2)
        logger.info(f"Saved {len(self.failed_docs)} failed docs to {config.failed_docs_path}")

    def _print_summary(self, summary: Dict):
        """Print processing summary."""
        logger.info("=" * 60)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total eligible documents: {summary['total_eligible']}")
        logger.info(f"Already processed (skipped): {summary['skipped']}")
        logger.info(f"Processed this run: {summary['processed']}")
        logger.info(f"Failed this run: {summary['failed']}")
        logger.info(f"Success rate: {summary['success_rate']}%")
        logger.info(f"Duration: {summary['duration_seconds']}s")
        logger.info("=" * 60)

    def cleanup(self):
        """Clean up resources."""
        self.llm_judge.cleanup()

    def backfill_authors(self, doc_id: str = None, limit: int = None) -> Dict:
        """
        Backfill author2concepts table from existing doc2concepts data.

        Args:
            doc_id: Optional single document ID to backfill
            limit: Optional limit on number of documents to process

        Returns:
            Summary statistics
        """
        start_time = time.time()

        # Get doc_ids to process
        if doc_id:
            doc_ids = [doc_id]
            logger.info(f"Backfilling single document: {doc_id}")
        else:
            doc_ids = self.sql_client.get_all_processed_doc_ids_list()
            logger.info(f"Found {len(doc_ids)} documents for backfill")

        if limit:
            doc_ids = doc_ids[:limit]
            logger.info(f"Limited to {limit} documents")

        total = len(doc_ids)
        processed = 0
        failed = 0
        total_upserts = 0

        logger.info("=" * 60)
        logger.info("BACKFILL AUTHOR2CONCEPTS")
        logger.info("=" * 60)

        for idx, did in enumerate(doc_ids, 1):
            try:
                # 1. Get author_node_ids from kg-docs-index
                doc = self.search_client.get_document(did)
                if not doc:
                    logger.warning(f"[{idx}/{total}] Document not found in AI Search: {did}")
                    failed += 1
                    continue

                author_node_ids = doc.get('author_node_ids', []) or []
                authors = self.search_client.filter_author_node_ids(author_node_ids)

                if not authors:
                    logger.debug(f"[{idx}/{total}] No resolved authors for {did}")
                    processed += 1
                    continue

                # 2. Get concepts for this doc from doc2concepts
                concepts = self.sql_client.get_doc_concepts(did)

                if not concepts:
                    logger.warning(f"[{idx}/{total}] No concepts found for {did}")
                    failed += 1
                    continue

                # 3. UPSERT into author2concepts
                upsert_count = 0
                for author_id in authors:
                    for concept_id in concepts:
                        self.sql_client.upsert_author2concept(author_id, concept_id)
                        upsert_count += 1

                total_upserts += upsert_count
                processed += 1
                logger.info(f"[{idx}/{total}] {did}: {len(authors)} authors × {len(concepts)} concepts = {upsert_count} upserts")

            except Exception as e:
                logger.error(f"[{idx}/{total}] Failed {did}: {e}")
                failed += 1

        elapsed = time.time() - start_time

        # Print summary
        logger.info("=" * 60)
        logger.info("BACKFILL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total documents: {total}")
        logger.info(f"Processed: {processed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total upserts: {total_upserts}")
        logger.info(f"Duration: {elapsed:.2f}s")
        logger.info("=" * 60)

        return {
            'total': total,
            'processed': processed,
            'failed': failed,
            'total_upserts': total_upserts,
            'duration_seconds': round(elapsed, 2)
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Map ScholarBank documents to concepts using LLM-as-Judge'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of documents to process'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--doc-id',
        type=str,
        default=None,
        help='Process only a specific document ID'
    )
    parser.add_argument(
        '--fill-authors',
        action='store_true',
        help='Backfill author2concepts table from existing doc2concepts data'
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    config.setup_logging(log_level)

    logger.info("=" * 60)
    logger.info("ScholarBank Sample-to-Concept Mapping")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"Configuration:")
    logger.info(f"  - Hybrid search top_k: {config.hybrid_search_top_k}")
    logger.info(f"  - LLM judge top_k: {config.llm_judge_top_k}")
    logger.info(f"  - Final top_k: {config.final_top_k}")
    logger.info(f"  - Max retries: {config.max_retries}")
    logger.info(f"  - Min valid concepts: {config.min_valid_concepts}")
    if args.limit:
        logger.info(f"  - Document limit: {args.limit}")
    if args.doc_id:
        logger.info(f"  - Single document: {args.doc_id}")
    logger.info("-" * 60)

    try:
        processor = DocumentProcessor()

        if args.fill_authors:
            # Backfill author2concepts from existing doc2concepts
            logger.info("Mode: BACKFILL AUTHOR2CONCEPTS")
            summary = processor.backfill_authors(doc_id=args.doc_id, limit=args.limit)
            processor.cleanup()
            return 0 if summary['failed'] == 0 else 1
        elif args.doc_id:
            # Process single document
            success = processor.process_document(args.doc_id)
            processor.cleanup()
            return 0 if success else 1
        else:
            # Process all eligible documents
            summary = processor.run(limit=args.limit)
            processor.cleanup()
            return 0 if summary['failed'] == 0 else 1

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
