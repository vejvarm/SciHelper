from utils import _fetch_arxiv_article, fetch_and_download_from_arxiv
from flags import ARTICLE_DIR

if __name__ == "__main__":
    # article = "https://arxiv.org/abs/2303.12755"
    article = "https://arxiv.org/format/2303.12793"
    fetch_and_download_from_arxiv(article, ARTICLE_DIR)