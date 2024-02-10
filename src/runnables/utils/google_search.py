from typing import Generator, Tuple
import requests
from logging import getLogger
from urllib import parse
from os.path import splitext

from googlesearch import search
import bs4


logger = getLogger(__name__)


def parse_url(url: str) -> Tuple[str, str]:
    logger.info(f"Parsing URL: {url}")
    parsed_url = parse.urlparse(url)
    _, ext = splitext(parsed_url.path)
    if ext not in [".html", ".htm", ""]:
        logger.warning(f"URL: {url} is not an HTML page")
        raise Exception(f"URL: {url} is not an HTML page")
    page_content = requests.get(url).text
    soup = bs4.BeautifulSoup(page_content, "html.parser")
    parsed_page_content = soup.get_text()
    page_title = soup.title
    page_title = str(page_title.string) if page_title else "Unknown Title"
    return page_title, parsed_page_content

def format_page_content(title: str, url: str, page_content: str) -> str:
    logger.debug(f"Formatting page content for URL: {url}, title: {title}, page content: {page_content[:100]}")
    return f"Title: {title}\nLink: {url}\nPage content: '''{page_content}'''"


def get_information_from_google_search(query: str) -> Generator[str, None, None]:
    logger.info(f"Searching for information on the web for query: {query}")
    for url in search(query, tld="co.in", num=10, stop=10, pause=2):
        try:
            title, page_content = parse_url(url)
            yield format_page_content(title, url, page_content)
        except Exception:
             logger.warning(f"Failed to parse URL: {url}")
        
