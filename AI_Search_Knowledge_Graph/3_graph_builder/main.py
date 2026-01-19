#!/usr/bin/env python3
"""
Main entry point for graph builder script.

Usage:
    python main.py                    # Run both parts
    python main.py --retry-first      # Run first part only (doc processing)
    python main.py --retry-second     # Run second part only (author interests)
    python main.py --doc-id DOC_ID    # Process single document
    python main.py --limit N          # Limit documents to process
    python main.py --debug            # Enable debug logging
"""

import argparse
import logging
import sys
from datetime import datetime
from typing import Dict, Optional

from config import config
from sql_client import SQLClient, StageStatus
from graph_builder import GraphBuilder
from author_interest_builder import AuthorInterestBuilder

logger = logging.getLogger(__name__)


def process_first_part(doc_id: Optional[str] = None, limit: Optional[int] = None) -> Dict:
    """
    Process documents through stages 0-6.

    Args:
        doc_id: Optional single document ID (UUID format)
        limit: Optional limit on documents to process

    Returns:
        Summary statistics
    """
    sql = SQLClient()
    builder = GraphBuilder()

    try:
        # Get pending documents
        pending = sql.get_pending_docs(max_stage=6)

        if doc_id:
            # Filter to specific document
            pending = [p for p in pending if p['row_id'] == doc_id]
            if not pending:
                # Check if doc exists but is already complete
                progress = sql.get_doc_progress(doc_id)
                if progress and progress['current_stage'] == 6 and progress['stage_status'] == 'completed':
                    logger.info(f"Document {doc_id} is already complete at stage 6")
                    return {'total': 0, 'completed': 0, 'failed': 0, 'skipped': 1}
                elif not progress:
                    logger.error(f"Document {doc_id} not found in asm_samples_progress")
                    return {'total': 0, 'completed': 0, 'failed': 1, 'skipped': 0}

        if limit:
            pending = pending[:limit]

        total = len(pending)
        completed = 0
        failed = 0

        logger.info(f"Processing {total} documents")
        logger.info("=" * 60)

        for idx, row in enumerate(pending, 1):
            row_id = row['row_id']
            current_stage = row['current_stage'] or 0
            retry_count = row['retry_count'] or 0

            logger.info(f"[{idx}/{total}] Processing {row_id} from stage {current_stage}")

            # Check retry limit
            if retry_count >= config.max_retries:
                logger.warning(f"Skipping {row_id}: exceeded retry limit ({retry_count})")
                failed += 1
                continue

            # Update status to in-progress
            sql.mark_stage_in_progress(row_id, current_stage)

            # Process document
            success, last_stage, err_msg = builder.process_document(row_id, current_stage)

            if success:
                sql.mark_stage_completed(row_id, last_stage)
                completed += 1
                logger.info(f"[{idx}/{total}] SUCCESS: {row_id}")
            else:
                sql.mark_stage_errored(row_id, last_stage, err_msg)
                failed += 1
                logger.error(f"[{idx}/{total}] FAILED: {row_id} at stage {last_stage} - {err_msg}")

        return {
            'total': total,
            'completed': completed,
            'failed': failed,
            'skipped': 0
        }
    finally:
        builder.close()


def process_second_part() -> Dict:
    """
    Process author research interests.

    Returns:
        Summary statistics
    """
    builder = AuthorInterestBuilder()

    try:
        total, created, errors = builder.build_research_interests()

        return {
            'total_mappings': total,
            'edges_created': created,
            'errors': errors
        }
    finally:
        builder.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Build knowledge graph in CosmosDB Gremlin',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./run.sh                         Run both parts (default)
  ./run.sh --retry-first           Run first part only (document processing)
  ./run.sh --retry-second          Run second part only (author interests)
  ./run.sh --doc-id abc123         Process single document
  ./run.sh --limit 10              Process only 10 documents
  ./run.sh --debug                 Enable debug logging
        """
    )
    parser.add_argument(
        '--retry-first',
        action='store_true',
        help='Run first part only (document processing stages 0-6)'
    )
    parser.add_argument(
        '--retry-second',
        action='store_true',
        help='Run second part only (author research interests)'
    )
    parser.add_argument(
        '--doc-id',
        type=str,
        default=None,
        help='Process only a specific document ID (UUID format, without doc_ prefix)'
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
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    config.setup_logging(log_level)

    logger.info("=" * 60)
    logger.info("Graph Builder - CosmosDB Gremlin")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"Target collection: {config.gremlin_collection}")

    if args.doc_id:
        logger.info(f"Single document mode: {args.doc_id}")
    if args.limit:
        logger.info(f"Document limit: {args.limit}")

    logger.info("-" * 60)

    exit_code = 0

    try:
        # Determine what to run
        run_first = not args.retry_second
        run_second = not args.retry_first

        # If both flags are set, run both (edge case)
        if args.retry_first and args.retry_second:
            run_first = True
            run_second = True

        # Part 1: Document processing
        if run_first:
            logger.info("")
            logger.info("PART 1: Document Processing (Stages 0-6)")
            logger.info("=" * 60)

            summary1 = process_first_part(doc_id=args.doc_id, limit=args.limit)

            logger.info("")
            logger.info("PART 1 SUMMARY:")
            logger.info(f"  Total documents: {summary1['total']}")
            logger.info(f"  Completed: {summary1['completed']}")
            logger.info(f"  Failed: {summary1['failed']}")
            if summary1.get('skipped'):
                logger.info(f"  Skipped (already complete): {summary1['skipped']}")

            if summary1['failed'] > 0:
                exit_code = 1

        # Part 2: Author research interests
        if run_second:
            logger.info("")
            logger.info("PART 2: Author Research Interests")
            logger.info("=" * 60)

            summary2 = process_second_part()

            logger.info("")
            logger.info("PART 2 SUMMARY:")
            logger.info(f"  Total mappings: {summary2['total_mappings']}")
            logger.info(f"  Edges created: {summary2['edges_created']}")
            logger.info(f"  Errors: {summary2['errors']}")

            if summary2['errors'] > 0:
                exit_code = 1

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Completed at: {datetime.now():%Y-%m-%d %H:%M:%S}")
        logger.info(f"Log file: {config.log_path}")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        exit_code = 130
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit_code = 1

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
