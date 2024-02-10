"""Web Crawler Runnable"""

from operator import itemgetter

from langchain_core.runnables import RunnablePassthrough

from .question_answering import question_answering_runnable
from .tavily_retriever import tavily_retriever


web_crawler_runnable = (
    RunnablePassthrough.assign(context=itemgetter("question") | tavily_retriever)
    | question_answering_runnable
)
