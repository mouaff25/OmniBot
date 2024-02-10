"""Search Query Runnable"""

from langchain.prompts import ChatPromptTemplate

from .utils.llm import get_llm

llm = get_llm()

search_query_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a search query system. Given the question, you should generate a search relevant search query. Only output the search query.",
        ),
        ("human", "{question}"),
    ]
)

search_query_runnable = search_query_prompt | llm
