from pathlib import Path
from flags import ROOT_DIR, ARTICLE_DIR
from utils import parse_pdf, split_into_sections, fetch_and_download_from_arxiv


if __name__ == "__main__":
    fetch_and_download_from_arxiv("2103.10385", ARTICLE_DIR)


    source_pdf_path = ROOT_DIR.joinpath("articles").glob("**/*.pdf")

    for p in source_pdf_path:
        text, metadata = parse_pdf(p)
        print(metadata)
        sections = split_into_sections(text)

        for section in sections:
            print(f"{section}")