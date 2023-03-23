import os
from pathlib import Path
from enum import Enum

ROOT_DIR = Path(__file__).parent.absolute()
ARTICLE_DIR = ROOT_DIR.joinpath("articles")


class NS(Enum):
    USER_INIT = 'user-init'
    ARTICLES = 'articles'
    CONVOS = 'convos'


N_TOP_CONVOS = 5
N_TOP_ARTICLES = 3