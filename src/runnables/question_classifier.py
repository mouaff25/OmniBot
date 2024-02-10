"""Question Classifier runnable"""

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from .utils.llm import get_llm

llm = get_llm()

question_classifier_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 'Output "yes" if the input is a question or a request, and "no" if it is not.'),
        ("human", "Input: {input}"),
    ]
)

question_classifier_runnable = question_classifier_prompt | llm
