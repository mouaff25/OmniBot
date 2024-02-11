""" Standalone Question runnable """

from typing import List
from operator import itemgetter

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage

from .utils.llm import get_llm

llm = get_llm()


def format_messages(messages: List[BaseMessage], last_n_messages: int = 5) -> str:
    return "\n".join(
        f"{message.type}: {message.content}"
        for message in messages[-min(len(messages), last_n_messages) :]
    )


standalone_question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Given the following conversation and a follow-up question, rephrase the question into a standalone question that is suitable for a search query. Only output the standalone question.
Make sure to add any relevant context from the conversation to the standalone question to make answering the question easier.
            
{history}

Follow-up question: {question}""",
        ),
    ]
)

standalone_question_runnable = (
    RunnablePassthrough.assign(history=itemgetter("history") | RunnableLambda(func=format_messages))
    | standalone_question_prompt
    | llm
)
