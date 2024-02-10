"""Input Classifier runnable"""

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from .utils.llm import get_llm

llm = get_llm()

input_classifier_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a classifier designed to classify the user's input into one of the following categories:
- question-answering: The user is asking a question that requires a factual answer that is only answerable by looking up information.
- recommender: The user is asking for a recommendation / suggestion.
- creative: The user is looking for creative assistance, brainstorming, coming up with inventions, etc.
- conversation: The user is looking for a conversation partner to chat with.
- other: The user's input does not fit into any of the above categories.
         
Only output the name of the category that best fits the user's input.
"""),
        ("human", "User input: {input}"),
    ]
)

input_classifier_runnable = input_classifier_prompt | llm
