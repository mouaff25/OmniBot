"""Recommender runnable"""

from langchain.prompts import ChatPromptTemplate

from .utils.llm import get_llm

llm = get_llm()

recommender_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a multi-purposed recommender system. You can recommend items based on the user's input from movies, books, music, and more. Given page contents from the web and an item suggestion request, recommend an item (or items) from the page contents.
Your answer should be in the following format:
<suggestion>

Source: [<title>](<link>)

If the web page contents are not enough to recommend an item, output "I don't have enough information to recommend an item.".

{context}
""",
        ),
        ("human", "Item suggestion request: {item}"),
    ]
)

recommender_runnable = recommender_prompt | llm
