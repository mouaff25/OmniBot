""" Answer Validation runnable """

from langchain.prompts import ChatPromptTemplate

from .utils.llm import get_llm

llm = get_llm()

def parse_llm_output(output: str) -> bool:
    return "yes" in output
    


validate_answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Given a user's input and an agent's thought process, output whether the agent has reached the correct answer. If the agent's last observation is a valid answer to the user's question output "yes". If the reasoning is incomplete output "no".
The agent's thought process will have the following format:
```
Thought: <thought> (This is used to show the agent's thought process)
Action: <action> (the name of the action to take)
Action Input: <action input> (the input to the action)
Observation: (the result of the action)
```
(This Thought, Action, Action Input, and Observation format could be repeated multiple times to show the thought process and the actions taken to reach the final response.)

The whole though process is given for reference. Output "yes" if the last observation is a valid answer to the user's input, otherwise output "no".

Begin!
""",
        ),
        ("human", "User's input: {input}\nAgent's thought process: {agent_scratchpad}"),
    ]
)

validate_answer_runnable = validate_answer_prompt | llm | parse_llm_output
