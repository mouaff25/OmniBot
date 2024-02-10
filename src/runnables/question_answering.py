"""RetrievalQA chain"""

from langchain.prompts import ChatPromptTemplate

from .utils.llm import get_llm

llm = get_llm()

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a question-answering system. Using the following web pages, answer the question. Make sure to include the source of the information in your answer.
The context always contains the relevant information to answer the question. Use the context and your knowledge to answer the question.
Your answer should be in the following format:
<answer>

Source: [<title>](<link>)

If you think the context is not enough to answer the question, output "I don't have enough information to answer the question.".

{context}
""",
        ),
        ("human", "Question: {question}"),
    ]
)

question_answering_runnable = question_answering_prompt | llm
