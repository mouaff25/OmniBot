""" This module contains the Tavily retriever runnable. """
from typing import List, Sequence

from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever
from langchain_community.document_transformers.long_context_reorder import (
    LongContextReorder,
)
from langchain_core.documents import Document

from src.config.settings import get_settings


def format_tavily_response(response: List[Document]):
    return "\n\n\n".join(
        f"Title: {doc.metadata['title']}\nLink: {doc.metadata['source']}\nPage content: '''{doc.page_content}'''"
        for doc in response
    )


def reorder_documents(documents: Sequence[Document]) -> List[Document]:
    return list(LongContextReorder().transform_documents(documents))


settings = get_settings()
tavily_retriever = (
    TavilySearchAPIRetriever(
        api_key=settings.TAVILY_API_KEY,
        search_depth="basic", # type: ignore
        k=15,
    )
    | reorder_documents
    | format_tavily_response
)
