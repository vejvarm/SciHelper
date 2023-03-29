from pathlib import Path
from flags import ROOT_DIR, ARTICLE_DIR
from utils import parse_and_split_latex_file


if __name__ == "__main__":
    # fetch_and_download_from_arxiv("2103.10385", ARTICLE_DIR)

    source_tex_path = ROOT_DIR.joinpath("articles").glob("**/main.tex")

    for pth in source_tex_path:
        print(pth)
        sections = parse_and_split_latex_file(pth)

        for section, content in sections.items():
             print(f"{section.center(50, '-')}\n{content.strip()}\n{''.center(50, '-')}\n\n")
