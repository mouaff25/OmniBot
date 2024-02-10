from operator import itemgetter

from langchain_core.runnables import RunnableBranch, RunnablePassthrough

from .web_crawler_agent import web_crawler_agent_runnable
from .conversation_runnable import conversation_runnable


conversation_runnable_2 = {
    "history": itemgetter("history"),
    "input": itemgetter("original_input"),
} | conversation_runnable

omnibot_runnable = RunnablePassthrough.assign(
    conversation_response=conversation_runnable
) | RunnableBranch(
    (
        lambda x: "I'm sorry, I'm not able to answer this question" in x["conversation_response"], # type: ignore
        web_crawler_agent_runnable,
    ),
    lambda x: x["conversation_response"], # type: ignore
)
