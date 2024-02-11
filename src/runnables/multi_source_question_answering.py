from operator import itemgetter
from functools import lru_cache
from logging import getLogger

from langchain_core.runnables import RunnableLambda

from .tavily_retriever import tavily_retriever
from .question_answering import question_answering_runnable
from .utils.google_search import get_information_from_google_search


logger = getLogger(__name__)

no_information_response = "I don't have enough information"

def answer_question(question: str, context: str) -> str:
    try:
        return question_answering_runnable.invoke(
            {"question": question, "context": context}
        )
    except Exception:
        logger.warning(f"Blocked prompt exception for question: {question} and context: {context[:100]}")
        return no_information_response

@lru_cache(maxsize=128)
def answer_question_from_multiple_sources(question: str):
    logger.info(f"Answering question from multiple sources: {question}")
    context = tavily_retriever.invoke(question)
    answer = answer_question(question, context)
    
    if no_information_response in answer:
        logger.warning(
            f"Answering question from multiple sources failed. Trying to get information from google seach for question: {question}"
        )
        for context in get_information_from_google_search(question):
            logger.debug(f"Using context: {context[:100]} to answer question: {question}")
            answer = answer_question(question, context)
            if no_information_response not in answer:
                return answer
    return answer


multi_source_question_answering_runnable = {
    "question": itemgetter("question")
} | RunnableLambda(func=lambda x: answer_question_from_multiple_sources(x["question"]))
