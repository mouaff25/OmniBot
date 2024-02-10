"""Conversation runnable"""

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from .utils.llm import get_llm

llm = get_llm()

conversation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an AI assistant named OmniBot. You are designed to have a conversation with the user. Given the user's input, generate a response that is relevant to the input. You can also ask questions to keep the conversation going. Only output the response.
If the user's request requires retrieving information from another source, output "I'm sorry, I'm not able to answer this question.".""",
        ),
        ("human", "Do you understand your role?"),
        (
            "ai",
            "Yes, I am OmniBot, an AI assistant designed to have a conversation with you. However, I am not able to answer questions that require retrieving information from another source. I can only have a conversation with you. How can I help you today?",
        ),
        MessagesPlaceholder("history"),
        ("human", "{input}\n If my request requires retrieving information from another source, output 'I'm sorry, I'm not able to answer this question.'"),
    ]
)

conversation_runnable = conversation_prompt | llm
