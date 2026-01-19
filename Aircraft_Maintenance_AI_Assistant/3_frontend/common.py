import hashlib
import os
from pathlib import Path
import pandas as pd
from loguru import logger

# For testing only - to see environ variable is availabe in environ
env_vars = list(os.environ.keys())
logger.info("*" * 20)
logger.info("Environment Variables")
logger.info(env_vars)
logger.info("*" * 20)

# Detect if the program is running in AWS or local
LOCALTEST = False  # Default is aws
found = False
for var in env_vars:
    if "SIA_ENV" in var:
        found = True
if not found:
    LOCALTEST = True
logger.info(f"SIA AWS environment variables found? = {found}")

base_path = Path(__file__).parent

################################################################################
# Haystack DB files
# model path
MODEL_FOLDER = base_path# / "model"

# data path
DATA_FOLDER = base_path / "input/data/training/data/hr"
DATA_FILE = DATA_FOLDER / "qa.xlsx"

# General Parameters
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
MIN_QUERY_LENGTH = 2
LEV_DISTANCE_THRESHOLD = 0.9  # compare query with kb question

# Access credentials
deptt = ['ESD','CSD','SUPPORT','ALL_DEPT','ADMIN', 'HR', 'TSP']
win2k = ['curie_poc', 'HR', 'TSP']

################################################################################
# Information Retrieval / Retrieve and Re-Rank Algorithm parameters

# Original models
# EMBEDDING_MODEL = 'distilroberta-base-msmarco-v2' # Original
# EMBEDDING_MODEL = 'msmarco-distilbert-dot-v5' # Seems to be better than orig mod
# EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1"
# EMBEDDING_LENGTH = 768
# EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
# EMBEDDING_LENGTH = 384

# CROSSENCODING_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # Original

# EMBEDDING_NUM_RESULTS = 10
RANKER_NUM_RESULTS = 5

saved_vector_db = "./just_values"
CURIE_OUTPUT_PATH = 'curie_output.xlsx'


# Min Information Retrieval score to include in OpenAI qury
# IR_ANSWER_THRESHOLD = 0.3
# IR_ANSWER_THRESHOLD = 0.0001

################################################################################
# GPT Parameters - for openai api call
GPT_MODEL_PARAMS = {
    "api_type": "azure",
    "api_base": os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com/"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    "deployment_name": os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT", "GPT4O")
}

MIN_GPT_QUERY_LENGTH = 2

# GPT Similarity Threshold score
GPT_ANSWER_THRESHOLD = 2

# GPT Prefix in result. Include any required symbols or characters
# GPT_ANSWER_PREFIX = "[GPT3 BETA] "
GPT_ANSWER_PREFIX = (
    "**Disclaimer:**\nPlease note that we are still in a testing phase "
    + "for Curie-GPT. You may want to validate the answer provided against "
    + "the cited sources.\n\n"
)


################################################################################
def get_hash(df_: pd.DataFrame, col: str = "hval"):
    """Get the hash value for the hval column in the dataframe

    :param df_: Dataframe with a column named hval. This column should contain the text.
    :type df_: pd.DataFrame
    :param col: Column name containing the text to be hashed, defaults to "hval"
    :type col: str, optional
    :return: Dataframe with the hash value added as a new column
    :rtype: pd.DataFrame
    """
    df_[col] = df_[col].apply(
        lambda x: hashlib.md5(x.encode()).hexdigest() if pd.notna(x) else ""
    )
    return df_
