from io import BytesIO

import requests
from pypdf import PdfReader


def arxiv_pdf_reader(
    url: str, page_range: int | list = None, by_page: bool = False
) -> str:
    """
    Extracts PDF text from an arXiv URL.

    Parameters:
        url (str): The URL of the arXiv PDF.
        range (list): The range of pages to extract text from given as a list of indexes (starting from 0).
        by_page (bool): If True, the function returns a dictionary with the text per page. Otherwise, it returns a single string (default).

    Returns:
        str: The extracted text from the specified page.
    """
    response = requests.get(url)

    if response.status_code == 200:
        pdf_file = BytesIO(response.content)
        reader = PdfReader(pdf_file)
        match page_range:
            case None:
                start = 0
                end = len(reader.pages)
            case int() as single_page:
                start = single_page
                end = single_page + 1
            case list() as page_list if len(page_list) == 2:
                start, end = page_list
            case _:
                raise ValueError("Invalid page_range type. Must be int, list, or None.")
        if not by_page:
            text = ""
            for page in range(start, end):
                text += reader.pages[page].extract_text()
        else:
            text = {}
            for page in range(start, end):
                text[page] = reader.pages[page]
        return text

    else:
        raise Exception(
            f"Failed to retrieve PDF from {url}. Error code: {response.status_code}"
        )
