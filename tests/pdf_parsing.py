from pathlib import Path
from flags import ROOT_DIR, ARTICLE_DIR
from utils import _parse_pdf, parse_pdf_file_and_split, fetch_and_download_from_arxiv


if __name__ == "__main__":
    # fetch_and_download_from_arxiv("2103.10385", ARTICLE_DIR)

    source_pdf_path = ROOT_DIR.joinpath("articles").glob("**/*.pdf")

    for pth in source_pdf_path:
        sections = parse_pdf_file_and_split(pth)

        for section, content in sections.items():
            print(f"{section.center(50, '-')}\n{content.strip()}\n{''.center(50, '-')}\n\n")