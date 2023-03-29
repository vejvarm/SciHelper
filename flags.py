from pathlib import Path
from enum import Enum

# USER FLAGS
GENERATE_SUMMARY = True

ROOT_DIR = Path(__file__).parent.absolute()
ARTICLE_DIR = ROOT_DIR.joinpath("articles")
NEXUS_DIR = ROOT_DIR.joinpath("nexus")
LOG_DIR = ROOT_DIR.joinpath("gpt3_logs")

TIME_FORMAT = "%Y-%M-%d_%H%M%S"

DEFAULT_EMBEDDING_ENGINE = 'text-embedding-ada-002'

class NS(Enum):
    """Pinecone namespaces"""
    USER_INIT = 'user-init'
    ARTICLES = 'articles'
    CONVOS = 'convos'


N_TOP_CONVOS = 5
N_TOP_SECTIONS = 3
N_MAX_WORDS_IN_CONV = 1500
