from operator import itemgetter
from logging import getLogger

from langchain_core.runnables import RunnableLambda
from google.generativeai.types.generation_types import BlockedPromptException

from .tavily_retriever import tavily_retriever
from .recommender_runnable import recommender_runnable
from .utils.google_search import get_information_from_google_search


logger = getLogger(__name__)

no_information_response = "I don't have enough information to recommend an item"

def answer_question(item: str, context: str) -> str:
    try:
        return recommender_runnable.invoke(
            {"item": item, "context": context}
        )
    except Exception:
        logger.warning(f"Blocked prompt exception for question: {item} and context: {context[:100]}")
        return no_information_response


def sugeest_from_multiple_sources(item: str):
    logger.info(f"Answering question from multiple sources: {item}")
    context = tavily_retriever.invoke(item)
    answer = answer_question(item, context)
    
    if no_information_response in answer:
        logger.warning(
            f"Answering question from multiple sources failed. Trying to get information from google seach for question: {item}"
        )
        for context in get_information_from_google_search(item):
            logger.debug(f"Using context: {context[:100]} to answer question: {item}")
            answer = answer_question(item, context)
            if no_information_response not in answer:
                return answer
    return answer


multi_source_recommender_runnable = {
    "item": itemgetter("item")
} | RunnableLambda(func=lambda x: sugeest_from_multiple_sources(x["item"]))
