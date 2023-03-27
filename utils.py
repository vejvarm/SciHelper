import json
import logging
import re
import os
import tarfile
import requests
import arxiv
import PyPDF2
from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode, LatexMacroNode, LatexWalkerError, LatexGroupNode
from pylatexenc.latex2text import LatexNodes2Text
from pathlib import Path
from flags import ROOT_DIR, ARTICLE_DIR
from typing import Sequence, Union, Callable

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.WARNING)


def cleanup_prompt(prompt: str):
    return prompt.encode(encoding='ASCII', errors='ignore').decode().strip()


def pineconize(article_id: str):
    """ as pinecone thinks that article names from arxiv are actually datetime objects, we need to replace dots with
    something else

    :param article_id: (str) id of article in the format \d{4}.\d{4,5}
    :return: (str) id of article with '.' replaced by '!'
    """

    return article_id.replace('.', '!')


def depineconize(pineconized_id: str):
    """
    Reverses the transformation applied by the pineconize function.

    :param pineconized_id: (str) id of an article with '!' instead of '.'
    :return: (str) the original article id with '.' instead of '!'
    """
    return pineconized_id.replace('!', '.')


def extract_text_from_completion_api_response(response: dict) -> str:
    text = response['choices'][0]['text'].strip()
    text = re.sub('[\r\n]+', '\n', text)
    return re.sub('[\t ]+', ' ', text)


def extract_text_from_chat_api_response(response: dict):
    role = response['choices'][0]['message']['role']
    text = response['choices'][0]['message']['content'].strip()
    text = re.sub('[\r\n]+', '\n', text)
    text = re.sub('[\t ]+', ' ', text)
    return role, text


def make_article_metadata(id: str, title="", authors: Sequence[str] = tuple("", ),
                          categories: Sequence[str] = tuple("", ), published="", journal_ref="",
                          links: Sequence[str] = tuple("", ), url="", summary="",
                          section_summaries: dict[str: tuple[str]] = None):
    """Helper function for creating metadata dictionary for json export. id field is mandatory as unique identifier.

    :param section_summaries: (dict[str: tuple[str]]) e.g. {vector_id: (section_title, section_content), }
    """
    return {"id": id.strip(),  # mandatory (ideally same as containing folder name)
            "title": str(title).strip(),
            "authors": tuple(authors),
            "categories": tuple(categories),
            "published": str(published).strip(),
            "journal_ref": str(journal_ref).strip(),
            "links": tuple(links),
            "url": str(url).strip(),
            "summary": str(summary).strip(),
            "section_summaries": dict() if section_summaries is None else section_summaries,   # Will be filled with GPT summaries and correspoding vector ids of sections
            }


def extract_tgz(source_archive: str or Path, extraction_folder: str or Path):
    """Extract """
    source = Path(source_archive)
    if not source.name.endswith(".tgz") and not source.name.endswith("tar.gz"):
        raise NotImplementedError(f"File {source.name} is not a '.tgz' file.")

    Path(extraction_folder).mkdir(exist_ok=True)

    try:
        tar = tarfile.open(source)
        tar.extractall(extraction_folder)
        tar.close()
    except tarfile.ReadError as err:
        LOGGER.warning(err)
        LOGGER.warning("No source files were extracted!")


def extract_arxiv_id(string):
    pattern = r"(?:(?:arXiv:)?([0-9]{2}[0-1][0-9]\.[0-9]{4,5}(?:v[0-9]+)?|(?:[a-z|\-]+(\.[A-Z]{2})?\/\d{2}[0-1][0-9]\d{3})))"
    match = re.search(pattern, string.strip())
    if match:
        return match.group(1)
    else:
        return None


def _fetch_arxiv_article(article_id: str):
    response = arxiv.Search(id_list=[article_id],
                            max_results=1,
                            sort_by=arxiv.SortCriterion.LastUpdatedDate,
                            sort_order=arxiv.SortOrder.Descending)
    if len(response.id_list) >= 1:
        return next(response.results())
    else:
        LOGGER.warning("There is no article with that id!")
        return None


def fetch_and_download_from_arxiv(article: str, article_folder: Path or str, sourcefiles_subfolder="tex") -> tuple[Path or None, dict]:
    article_id = extract_arxiv_id(article)
    paper = _fetch_arxiv_article(article_id)
    metadata = make_article_metadata(id=article_id,
                                     title=paper.title,
                                     authors=[a.name for a in paper.authors],
                                     categories=paper.categories,
                                     published=paper.published.strftime("%Y-%M-%d"),
                                     journal_ref=paper.journal_ref,
                                     links=[l.href for l in paper.links],
                                     url=paper.pdf_url,
                                     summary=paper.summary)
    specific_folder = article_folder.joinpath(article_id)
    specific_folder.mkdir(exist_ok=True)
    json.dump(metadata, specific_folder.joinpath("metadata.json").open("w"), indent=4)
    LOGGER.info(f"Downloading pdf from ArXiv for article with id: {article_id}.\nThis may take a while ...")
    paper.download_pdf(dirpath=str(specific_folder), filename="paper.pdf")
    paper.download_source(dirpath=str(specific_folder), filename="paper.tgz")
    LOGGER.debug(specific_folder.joinpath("paper.tgz"))
    extract_tgz(specific_folder.joinpath("paper.tgz"), specific_folder.joinpath(sourcefiles_subfolder))
    LOGGER.info("... download complete")

    main_tex_file_path = specific_folder.joinpath(sourcefiles_subfolder).joinpath('main.tex')
    all_tex_paths = list(specific_folder.joinpath(sourcefiles_subfolder).glob("**/*.tex"))
    pdf_file_path = specific_folder.joinpath("paper.pdf")

    if main_tex_file_path.exists():
        LOGGER.debug("returning path to 'main.tex'")
        return main_tex_file_path, metadata
    elif all_tex_paths:
        for tex_path in all_tex_paths:
            try:
                _, _ = _parse_latex_file(tex_path)
                LOGGER.info(f"returning path to '{tex_path}'")
                return tex_path, metadata
            except LatexWalkerError as err:
                LOGGER.debug(f"{err} no document node found in {tex_path}")
        LOGGER.warning("No 'document' nodes found in source tex files (most likely no .tex source files were fetched)")
    elif pdf_file_path.exists():
        LOGGER.debug("returning path to 'paper.pdf'")
        return pdf_file_path, metadata
    else:
        LOGGER.warning("No acceptable files (pdf or tex) for parsing were fetched.")
        return None, metadata


def _resolve_auxilary_tex_files(source_text: str, source_folder: Path) -> str:
    """ Search through given text and find all instances of '\input{NAME}'.
    For each instance of '\input{NAME}' found, look for file called 'NAME.tex' and replace the '\input{NAME}' entry
    with the contents of the 'NAME.tex' file.
    :param source_text: raw text string from the 'main.tex' file
    :param source_folder: Path to the source folder of the main.tex file
    :return parsed_text: source_text with all found instances of '\input{NAME}' replaced with the contents of their respective NAME.tex files
    """
    pattern = r"\\input\{(.+?)(?:\.tex)?\}"  # TODO: reduce the . to only allowed characters in filename strings
    matches = re.findall(pattern, source_text)
    for match in matches:
        tex_file = match + '.tex'
        matched_pattern = '\\input{' + match + '}'
        try:
            with source_folder.joinpath(tex_file).open('r') as f:
                tex_contents = f.read()
        except FileNotFoundError as er:
            LOGGER.warning(f"{er}\nsearching recursively through subfolders")

            try:
                file = next(source_folder.glob(f"**/{tex_file}"))
                with file.open('r') as f:
                    tex_contents = f.read()
            except StopIteration:
                LOGGER.warning(f'No file with name {tex_file} found, keeping the original {matched_pattern} clause')
                tex_contents = matched_pattern

        source_text = source_text.replace(matched_pattern, tex_contents)

    return source_text


def choose_source_file_parser(source_file: Union[Path, str]) -> Callable:
    """Validate if arXiv source file type is compatible with article parser and return appropriate parser function.

    The supported file types from arXiv are .tex and .pdf.
    - if .tex, return split_parsed_latex_file()
    - if .pdf, return parse_pdf_file_and_split()

    :param source_file: path to arXiv source file (main.tex or {article_id}.pdf)
    :return: function to parse source file
    :raises ValueError: if file extension is not .tex or .pdf
    """
    source_file = Path(source_file)
    if source_file.suffix == ".tex":
        return split_parsed_latex_file
    elif source_file.suffix == ".pdf":
        return parse_pdf_file_and_split
    else:
        raise ValueError("Unsupported file type. Only .tex and .pdf files are supported.")


def _read_text_from_file(source_tex_file: Path or str) -> str:
    source_file = Path(source_tex_file)
    try:
        with source_file.open() as f:
            return f.read()
    except FileNotFoundError:
        LOGGER.warning(f"Specified file at path {source_file} doesn't exist. Returning empty string.")
        return ''


def _parse_latex_file(source_tex_file: Path or str):
    source_file = Path(source_tex_file)
    text = _read_text_from_file(source_file)

    # parse auxilary .tex files which are referenced by /input{} in source_tex_file
    text = _resolve_auxilary_tex_files(text, source_file.parent)
    LOGGER.debug(text)

    walker = LatexWalker(text)
    nodelist, _pos, _len = walker.get_latex_nodes()

    # find 'document' EnvironmentNode
    for node in nodelist:
        if node.isNodeType(LatexEnvironmentNode) and node.environmentname == 'document':
            doc_nodelist = node.nodelist
            break
    else:
        raise LatexWalkerError("No document node found in provided tex file. Is it a full latex document?")  # TODO: catch

    abstract_section = re.compile(r'\\begin\{abstract\}((.|\n)+)\\end\{abstract\}')
    sections = {'abstract': re.search(abstract_section, text).group(0)}

    return sections, doc_nodelist


def split_parsed_latex_file(source_tex_file: Path or str) -> dict[str: str]:
    """ Function that parses a provided latex document file and splits it by \section{...} nodes of the document
    :param tex: path to main LaTeX file of the article. Usually should be called main.tex
    :return sections (dict[section name: section content]): section names and contents parsed in plain text (unicode)
    """
    # DONE (critical): add support for parsing `\input{1_intro}` files into `main.tex`
    # TODO (normal): add support for citations (currently the parsed content has `<cit.>` everywhere)
    # return doc_nodelist, sections,
    # latex2text
    latex2text = LatexNodes2Text(math_mode='text',  # unicode TODO: test if verbatim better
                                 fill_text=False,
                                 strict_latex_spaces=True)  # TODO: replace excessive /n characters
    current_content = []
    current_section = None
    sections, doc_nodelist = _parse_latex_file(source_tex_file)

    for node in doc_nodelist:
        LOGGER.debug(f"node@for node in doc_nodelist: {node}")
        if node.isNodeType(LatexMacroNode) and node.macroname == 'section':
            LOGGER.debug(f"node@if node.isNodeType(LatexMacroNode) and node.macroname == 'section': {node}")
            if current_section is not None:
                sections[current_section.strip()] = ''.join(latex2text.nodelist_to_text(current_content)).strip()
            for n in node.nodeargd.argnlist:
                LOGGER.debug(f"n@for n in node.nodeargd.argnlist: {n}")
                if n is None:
                    continue
                if n.isNodeType(LatexGroupNode):
                    LOGGER.debug(f"n@for n in node.nodeargd.argnlist: {n}")
                    current_section = n.nodelist[0].chars
            current_content = []
        elif current_section is not None:
            current_content.append(node)
    if current_section is not None:
        sections[current_section.strip()] = ''.join(latex2text.nodelist_to_text(current_content)).strip()
    return sections


def _parse_pdf(source_pdf_file: Path or str) -> (str, PyPDF2.DocumentInformation):
    """Load pdf from local or remote source and parse it into plain text."""
    s = Path(source_pdf_file)

    if s.exists():
        pdf_file = s
    else:
        pdf_file = Path("temp.pdf")
        try:
            # Download the PDF from the URLarticle_folder.joinpath(f"{article_id}/metadata.json")
            response = requests.get(url=source_pdf_file)
            with pdf_file.open("wb") as f:
                f.write(response.content)
        except requests.RequestException as ex:
            LOGGER.warning(f"{ex}, returning ('', None)")
            return '', None

    # Open the PDF and parse the text
    with open(pdf_file, "rb") as f:
        LOGGER.info("Extracting text from PDF file (may take a while)...")
        pdf = PyPDF2.PdfReader(f)
        metadata = pdf.metadata
        text = ""
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            text += page.extract_text()

    # Clean up the temporary file
    try:
        os.remove("temp.pdf")
    except FileNotFoundError:
        LOGGER.info("temp.pdf didn't exist. Cleanup not necessary.")

    return text, metadata


def parse_pdf_file_and_split(source_pdf_file: Path or str) -> dict[str: str]:
    """Split parsed pdf (plain text) into parts, which correspond to Sections of the article"""
    # TODO: it doesn't do a good job --> REWRITE
    LOGGER.warning("Parsing from PDF is buggy and inefficient. "
                   "It will most likely have undesirable results.\n"
                   "Check if the provided article has available .tex source files on arXiv \n"
                   " --> look for 'Other formats' under 'PDF' Download section on the article page. \n"
                   "If Source (.tar.gz) are available, there is other problem, please report in Issues.")
    source_file = Path(source_pdf_file)
    text, metadata = _parse_pdf(source_file)

    section_pattern = re.compile(r'^\s*(?P<section>[A-Z][^\n]+)[\n\r]+', re.MULTILINE)
    matches = section_pattern.finditer(text)
    sections = dict()
    for match in matches:
        section_title = match.group('section')
        section_text = text[match.end():] if match.end() < len(text) else ''
        next_match = section_pattern.search(section_text) if section_text else None
        section_end = next_match.start() if next_match else len(section_text)
        section_content = section_text[:section_end].strip()
        sections[section_title.strip()] = section_content.strip()
    return sections
