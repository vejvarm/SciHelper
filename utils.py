import logging
import os
import requests
import PyPDF2
from pathlib import Path


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.WARNING)


def parse_pdf(source: Path or str):
    """"""
    s = Path(source)

    if s.exists():
        pdf_file = s
    else:
        pdf_file = Path("temp.pdf")
        try:
            # Download the PDF from the URL
            response = requests.get(url=source)
            with pdf_file.open("wb") as f:
                f.write(response.content)
        except requests.RequestException as ex:
            print(ex)
            return None, None

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