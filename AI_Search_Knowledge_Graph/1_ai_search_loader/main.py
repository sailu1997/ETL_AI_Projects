#!/usr/bin/env python3
"""Main orchestrator for loading ScholarBank data to Azure AI Search.

Usage:
    python main.py [--limit N]

Arguments:
    --limit N   Only process N documents (default: all eligible documents)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

from azure_clients import SQLClient, BlobClient, SearchClientWrapper
from index_manager import ensure_indexes_exist
from data_loader import DataLoader

# Script directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CRED_PATH = PROJECT_ROOT / '.cred.json'
SCHEMA_PATH = SCRIPT_DIR / 'aisearch_schema.json'
LOG_FILE = SCRIPT_DIR / 'sb_load_data.log'


def setup_logging() -> logging.Logger:
    """Setup logging to both console and file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)

    # File handler
    file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def load_credentials(cred_path: Path) -> Dict:
    """Load credentials from JSON file."""
    if not cred_path.exists():
        raise FileNotFoundError(f"Credentials file not found: {cred_path}")

    with open(cred_path, 'r') as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Load ScholarBank data to Azure AI Search'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit the number of documents to process'
    )
    parser.add_argument(
        '--recreate-indexes',
        action='store_true',
        help='Delete and recreate all indexes (use if schema changed)'
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Starting ScholarBank to AI Search loader")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    if args.limit:
        logger.info(f"Document limit: {args.limit}")
    if args.recreate_indexes:
        logger.info("Recreate indexes: YES (will delete and recreate all indexes)")
    logger.info("=" * 60)

    # Load credentials
    try:
        credentials = load_credentials(CRED_PATH)
        logger.info("Credentials loaded successfully")
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"Invalid credentials JSON: {e}")
        return 1

    # Validate required credentials
    required_creds = ['azure_sql', 'azure_blob', 'asm_ai_search']
    for cred in required_creds:
        if cred not in credentials:
            logger.error(f"Missing required credential: {cred}")
            return 1

    # Initialize clients
    try:
        sql_client = SQLClient(credentials['azure_sql'])
        blob_client = BlobClient(credentials['azure_blob'])
        search_client = SearchClientWrapper(credentials['asm_ai_search'])
        logger.info("Azure clients initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        return 1

    # Ensure indexes exist
    logger.info("Ensuring indexes exist...")
    if not ensure_indexes_exist(search_client.index_client, SCHEMA_PATH, args.recreate_indexes):
        logger.error("Failed to ensure indexes exist")
        return 1
    logger.info("All indexes are ready")

    # Get eligible documents
    logger.info("Fetching eligible documents from SQL...")
    try:
        eligible_docs = sql_client.get_eligible_documents(limit=args.limit)
    except Exception as e:
        logger.error(f"Failed to fetch eligible documents: {e}")
        return 1

    total_docs = len(eligible_docs)
    logger.info(f"Found {total_docs} eligible documents to process")

    if total_docs == 0:
        logger.info("No documents to process. Exiting.")
        return 0

    # Initialize data loader
    data_loader = DataLoader(sql_client, blob_client, search_client)

    # Process documents
    successful = 0
    failed = 0
    failed_docs = []

    for i, doc in enumerate(eligible_docs, 1):
        doc_id = doc['row_id']
        logger.info(f"[{i}/{total_docs}] Processing document: {doc_id}")

        try:
            success, message = data_loader.process_document(doc_id)
            if success:
                successful += 1
                logger.info(f"[{i}/{total_docs}] SUCCESS: {doc_id}")
            else:
                failed += 1
                failed_docs.append((doc_id, message))
                logger.error(f"[{i}/{total_docs}] FAILED: {doc_id} - {message}")
        except Exception as e:
            failed += 1
            failed_docs.append((doc_id, str(e)))
            logger.error(f"[{i}/{total_docs}] EXCEPTION: {doc_id} - {e}")

    # Print summary
    logger.info("=" * 60)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total documents: {total_docs}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {successful/total_docs*100:.1f}%")

    if failed_docs:
        logger.info("")
        logger.info("Failed documents:")
        for doc_id, reason in failed_docs:
            logger.info(f"  - {doc_id}: {reason}")

    logger.info("=" * 60)
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("=" * 60)

    # Return exit code based on results
    if failed > 0:
        return 1 if successful == 0 else 0  # Partial success is still success
    return 0


if __name__ == '__main__':
    sys.exit(main())
