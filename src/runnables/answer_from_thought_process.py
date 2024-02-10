""" Answer from thought process runnable """

from langchain.prompts import ChatPromptTemplate

from .utils.llm import get_llm

llm = get_llm()


answer_from_thought_process_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Given a user's question and an agent's thought process, output the final answer to the user's question. Even if the reasoning is incomplete, output the final answer.
The agent's thought process will have the following format:
```
Thought: <thought> (This is used to show the agent's thought process)
Action: <action> (the name of the action to take)
Action Input: <action input> (the input to the action)
Observation: (the result of the action)
```
(This Thought, Action, Action Input, and Observation format could be repeated multiple times to show the thought process and the actions taken to reach the final response.)

The whole though process is given for reference. Output the final answer to the user's question.

Begin!
""",
        ),
        ("human", "User's question: {input}\nAgent's thought process: {agent_scratchpad}"),
    ]
)

answer_from_thought_process_runnable = answer_from_thought_process_prompt | llm
