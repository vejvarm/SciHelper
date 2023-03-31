import json
import logging
import re
import os
import requests
import tarfile
from datetime import datetime

import arxiv
import numpy as np
import PyPDF2
import tiktoken
from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode, LatexMacroNode, LatexWalkerError, LatexGroupNode
from pylatexenc.latex2text import LatexNodes2Text
from pathlib import Path
from typing import Sequence, Union, Callable
from flags import Commands, N_MAX_TOKENS_IN_ARTICLE_SECTION

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.WARNING)


def handle_flag(conversation: list[dict[str:str, str:str]], expected_value: Commands) -> bool:
    message = conversation[-1]["content"]
    if expected_value.value in message:
        return True
    else:
        return False


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
                                     published=paper.published.strftime("%Y-%m-%d"),
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


def _resolve_input_statements_in_tex_file(source_text: str, source_folder: Path) -> str:
    """ Search through given text and find all instances of '\input{NAME}'.
    For each instance of '\input{NAME}' found, look for file called 'NAME.tex'/'NAME.tikz' and replace the
    '\input{NAME}' entry with the contents of the 'NAME.tex/.tikz' file.
    :param source_text: raw text string from the 'main.tex' file
    :param source_folder: Path to the source folder of the main.tex file
    :return parsed_text: source_text with all found instances of '\input{NAME}' replaced with the contents of their respective NAME.tex files
    """
    pattern = r"\\input\{(.+?)(?:\.tex)?\}"
    matches = re.findall(pattern, source_text)
    for match in matches:
        if match.endswith(".tikz"):
            file_name = match
        else:
            file_name = match + '.tex'
        matched_pattern = '\\input{' + match + '}'
        try:
            with source_folder.joinpath(file_name).open('r') as f:
                tex_contents = f.read()
        except FileNotFoundError as er:
            LOGGER.warning(f"{er}\nsearching recursively through subfolders")

            try:
                file = next(source_folder.glob(f"**/{file_name}"))
                with file.open('r') as f:
                    tex_contents = f.read()
            except StopIteration:
                LOGGER.warning(f'No file with name {file_name} found, keeping the original {matched_pattern} clause')
                tex_contents = matched_pattern

        source_text = source_text.replace(matched_pattern, tex_contents)

    return source_text


def choose_source_file_parser(source_file: Union[Path, str]) -> Callable:
    """Validate if arXiv source file type is compatible with article parser and return appropriate parser function.

    The supported file types from arXiv are .tex and .pdf.
    - if .tex, return parse_and_split_latex_file()
    - if .pdf, return parse_pdf_file_and_split()

    :param source_file: path to arXiv source file (main.tex or {article_id}.pdf)
    :return: function to parse source file
    :raises ValueError: if file extension is not .tex or .pdf
    """
    source_file = Path(source_file)
    if source_file.suffix == ".tex":
        return parse_and_split_latex_file
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
    text = _resolve_input_statements_in_tex_file(text, source_file.parent)
    LOGGER.debug(text)

    walker = LatexWalker(text)
    nodelist, _pos, _len = walker.get_latex_nodes()

    # find 'document' EnvironmentNode
    for node in nodelist:
        if node.isNodeType(LatexEnvironmentNode) and node.environmentname == 'document':
            doc_nodelist = node.nodelist
            break
    else:
        raise LatexWalkerError("No document node found in provided tex file. Is it a full latex document?")

    abstract_section = re.compile(r'\\begin\{abstract\}((.|\n)+)\\end\{abstract\}')
    sections = {'abstract': re.search(abstract_section, text).group(0)}

    return sections, doc_nodelist


def parse_and_split_latex_file(source_tex_file: Path or str) -> dict[str: str]:
    """ Function that parses a provided latex document file and splits it by \section{...} nodes of the document
    :param tex: path to main LaTeX file of the article. Usually should be called main.tex
    :return sections (dict[section name: section content]): section names and contents parsed in plain text (unicode)
    """

    latex2text = LatexNodes2Text(math_mode='text',
                                 fill_text=False,
                                 strict_latex_spaces=True)
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
        elif node.isNodeType(LatexEnvironmentNode) and node.environmentname == 'section':
            current_section = node.nodelist[0].nodelist[0].chars
            for node in node.nodelist[1:]:
                current_content.append(node)
            if current_section is not None:
                sections[current_section.strip()] = ''.join(latex2text.nodelist_to_text(current_content)).strip()
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


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def timestamp_to_datetime(unix_time):
    return datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")


def num_tokens_in_convo(convo: list[dict[str:str, str:str]], model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of conversations."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        LOGGER.warning("model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        LOGGER.warning("gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_in_convo(convo, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        LOGGER.warning("gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_in_convo(convo, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_in_convo() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how convo are converted to tokens.""")
    num_tokens = 0
    for message in convo:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def split_text_to_n_token_chunks(text: str, n_max_tokens: int = N_MAX_TOKENS_IN_ARTICLE_SECTION,
                                 avg_n_chars_per_token: int = 4) -> list[str]:
    """
    Splits a given text into a list of chunks, where each chunk has similar length and the length of each chunk does not exceed the given maximum number of tokens.

    :param text: The text to be split into chunks.
    :param n_max_tokens: The maximum number of tokens allowed in each chunk. Defaults to the value of N_MAX_TOKENS_IN_ARTICLE_SECTION.
    :param avg_n_chars_per_token: The average number of characters per token. Defaults to 4.
    :return: A list of chunks, where each chunk has similar length and the length of each chunk does not exceed the given maximum number of tokens.
    """
    num_chars = len(text)
    num_tokens = num_chars//avg_n_chars_per_token + 1
    LOGGER.info(f"approximate number of tokens: {num_tokens}")
    chunks = []
    n_chunks = num_tokens//n_max_tokens + 1

    split_positions = np.linspace(0, num_chars, n_chunks + 1, dtype=np.int64)
    for i in range(n_chunks):
        chunks.append(text[split_positions[i]:split_positions[i + 1]])

    return chunks
