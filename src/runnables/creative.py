"""Creative runnable"""

from langchain.prompts import ChatPromptTemplate

from .utils.llm import get_llm

llm = get_llm()

creative_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant designed to help with creative tasks. You can help with brainstorming, creative writing, coming up with ideas, inventions, etc. Given a user's input, generate a creative response that is relevant to the input.",
        ),
        ("human", "User input: {input}"),
    ]
)

creative_runnable = creative_prompt | llm
