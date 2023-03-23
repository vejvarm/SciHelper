from pathlib import Path
from flags import ROOT_DIR, ARTICLE_DIR
from utils import parse_latex_file, parse_latex_file_and_split


if __name__ == "__main__":
    source_tex_path = ROOT_DIR.joinpath("articles").glob("**/main.tex")

    # for p in source_tex_path:
    #     abstract, section = parse_latex_file(p)
    #
    #     print(f"Abstract:\n {abstract}")
    #     for heading, text in section:
    #         print(f"--------------------\n{heading}\n{text}\n-----------------\n\n")
    for pth in source_tex_path:
        print(pth)
        # print(text)
        sections = parse_latex_file_and_split(pth)

        for section, content in sections:
            print(f"{section.center(50, '-')}\n{content.strip()}\n{''.center(50, '-')}\n\n")
