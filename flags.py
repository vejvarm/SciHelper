from pathlib import Path
from enum import Enum

# USER FLAGS
GENERATE_SUMMARY = True

ROOT_DIR = Path(__file__).parent.absolute()
ARTICLE_DIR = ROOT_DIR.joinpath("articles")
TEMPLATES_DIR = ROOT_DIR.joinpath("templates")
NEXUS_DIR = ROOT_DIR.joinpath("nexus")
LOG_DIR = ROOT_DIR.joinpath("gpt3_logs")

TIME_FORMAT = "%Y-%m-%d_%H%M%S"

DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_EMBEDDING_ENGINE = 'text-embedding-ada-002'

class NS(Enum):
    """Pinecone namespaces"""
    USER_INIT = 'user-init'
    ARTICLES = 'articles'
    CONVOS = 'convos'


class Commands(Enum):
    """User text input commands for controlling the system"""
    EXIT = "exit program"               # terminates the main loop and exits the program
    RESET_CONVO = "reset conversation"  # terminates the current conversation and initializes a new conversation
    ADD_ARTICLE = "add article"         # initializes new article and adds its summary to current conversation

N_TOP_CONVOS = 5
N_TOP_SECTIONS = 3
CONVO_SCORE_CUTOFF = 0.5    # semantic search (vector) minimum similarity to fetch the result for previous conversations
SECTION_SCORE_CUTOFF = 0.8  # semantic search (vector) minimum similarity to fetch the result for article sections
N_MAX_WORDS_IN_CONV = 1500
N_MAX_TOKENS_IN_ARTICLE_SECTION = 2000
