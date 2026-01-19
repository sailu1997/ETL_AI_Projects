"""Configuration loader for sample-to-concept mapping."""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration for the sample-to-concept mapping script."""

    def __init__(self):
        self.script_root = Path(__file__).parent
        self.creds = self._load_credentials()

        # Azure SQL Configuration
        sql_creds = self.creds["azure_sql"]
        self.sql_server = sql_creds["server"]
        self.sql_database = sql_creds["database"]
        self.sql_username = sql_creds["username"]
        self.sql_password = sql_creds["password"]

        # Azure AI Search Configuration
        search_creds = self.creds["asm_ai_search"]
        self.search_endpoint = search_creds["endpoint"]
        self.search_api_key = search_creds["api_key"]
        self.concept_index = search_creds["concept_index_name"]  # kg-concepts-index
        self.doc_index = search_creds["doc_index_name"]  # kg-docs-index
        self.search_api_version = search_creds["api_version"]

        # Azure Foundry (LLM) Configuration
        foundry_creds = self.creds["azure_foundry"]
        self.foundry_endpoint = foundry_creds["endpoint"]
        self.foundry_api_key = foundry_creds["api_key"]
        self.foundry_deployment = foundry_creds["deployment_name"]
        self.foundry_api_version = foundry_creds["api_version"]

        # Processing Parameters
        self.hybrid_search_top_k = 50  # Initial candidates from hybrid search
        self.llm_judge_top_k = 20      # LLM evaluates all 50, returns top 20
        self.final_top_k = 15          # Keep only top 15 for storage

        # Retry Configuration
        self.max_retries = 5
        self.min_valid_concepts = 15
        self.retry_prompt_change_threshold = 3
        self.llm_timeout = 300  # 5 minutes

        # Output paths
        self.failed_docs_path = self.script_root / "failed_docs.json"
        self.log_path = self.script_root / "processing.log"

    def _load_credentials(self) -> Dict[str, Any]:
        """Load credentials from var.json."""
        cred_path = self.script_root / "var.json"
        if not cred_path.exists():
            raise FileNotFoundError(f"Credentials file not found: {cred_path}")
        with open(cred_path, 'r') as f:
            return json.load(f)

    def setup_logging(self, level=logging.INFO):
        """Configure console + file logging."""
        logger = logging.getLogger()
        logger.setLevel(level)

        # Clear existing handlers
        logger.handlers.clear()

        # Console handler with color
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(self.log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        return logger


# Global config instance
config = Config()
