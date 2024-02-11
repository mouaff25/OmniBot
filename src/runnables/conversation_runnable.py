"""Conversation runnable"""

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from .utils.llm import get_llm

llm = get_llm()

conversation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an AI assistant named OmniBot. You are designed to have a conversation with the user. Given the user's input, generate a response that is relevant to the input. You can also ask questions to keep the conversation going. Only output the response."""
        ),
        ("human", "Do you understand your role?"),
        (
            "ai",
            "Yes, I am OmniBot, an AI assistant designed to have a conversation with you. I can answer questions, retrieve information from the web, and generate creative ideas. I can also provide in-depth explanations and discussions on a wide range of topics. I'm here to help you!",
        ),
        MessagesPlaceholder("history"),
        ("human", "{input}\n If my request requires retrieving information from another source, output 'I'm sorry, I'm not able to answer this question.' Else, only answer me without paying attention to what I mentioned."),
    ]
)

conversation_runnable = conversation_prompt | llm
