from typing import List, Tuple
from operator import itemgetter

from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain.agents import AgentExecutor

from .multi_source_question_answering import multi_source_question_answering_runnable
from .multi_source_recommender import multi_source_recommender_runnable
from .standalone_question import standalone_question_runnable
from .creative import creative_runnable
from .validate_answer import validate_answer_runnable
from .answer_from_thought_process import answer_from_thought_process_runnable
from .utils.llm import _get_llm


def get_answer_from_web(question: str) -> str:
    """Get an answer from the web"""
    return f"```{multi_source_question_answering_runnable.invoke({'question': question})}```"


def get_suggestion_from_web(item: str) -> str:
    """Get a suggestion"""
    return f"```{multi_source_recommender_runnable.invoke({'item': item})}```"


def generate_creative_idea(input: str) -> str:
    """Generate a creative idea"""
    return f"```{creative_runnable.invoke({'input': input})}```"


def get_last_answer(intermediate_steps: List[Tuple[str, str]]):
    last_observation = intermediate_steps[-1][1].strip()[3:-3]
    return "I have the final answer.\nFinal Answer: " + last_observation


def format_last_answer(last_answer: str) -> str:
    return "I have the final answer.\nFinal Answer: " + last_answer


web_crawler_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a powerful assistant named OmniBot designed to be a able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, OmniBot is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand. OmniBot uses the tools at its disposal to provide the best possible response to the question it's given.
Whenever OmniBot is tasked with solving a problem, it breaks it down into simpler steps to use the tools at its disposal efficiently.
If you are asked about what you are capable of, you can introduce yourself as the state-of-the-art assistant OmniBot.
Make sure that your responses are helpful, detailed, informative, and relevant to the question at hand.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: <your thought here> (Use this to explain your thought process)
Action: <action> (the name of the action to take, should be one of [{tool_names}])
Action Input: <action input> (the input to the action)
Observation: (the result of the action)
```
(This Thought, Action, Action Input, and Observation format could be repeated as needed to show the thought process and the actions taken to reach the final response.)

When you have a final response you MUST use the format:

```
Thought: <your thought here> (Use this to explain your thought process)
Final Answer: <your response here> (Should be a detailed and informative response to the question)
```

Begin!""",
        ),
        ("human", """User input: {input}"""),
        ("ai", "{agent_scratchpad}"),
        ("human", ""),
    ],
)
tools: List[Tool] = [
    Tool(
        name="Get answer from web",
        func=get_answer_from_web,
        description="Use this tool to get an answer from the web, make sure that the question is simple, clear and not personal. It should not be too broad and should not contain keywords only, it should be a complete and clear question ending with a question mark.",
    ),
    Tool(
        name="Get suggestion from web",
        func=get_suggestion_from_web,
        description="Use this tool to get a suggestion based on information from the web, make sure that the item is simple, clear and not personal. The item could be anything from movies, books, music, etc.",
    ),
    Tool(
        name="Generate creative idea",
        func=generate_creative_idea,
        description="Use this tool to generate a creative idea, this could be anything from brainstorming, creative writing, coming up with ideas, inventions, etc.",
    ),
]
max_iterations = 3
prompt = web_crawler_agent_prompt.partial(
    tools=render_text_description(list(tools)),
    tool_names=", ".join([t.name for t in tools]),
    max_iterations=max_iterations,
)
llm_with_stop = _get_llm().bind(stop=["\nObservation:"])
agent = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
    )
    | RunnablePassthrough.assign(
        reached_answer=lambda x: (
            validate_answer_runnable.invoke(x) if x["intermediate_steps"] else False
        )
    )
    | RunnableBranch(
        (
            lambda x: x["reached_answer"], # type: ignore
            lambda x: get_last_answer(x["intermediate_steps"]),
        ),
        (
            lambda x: len(x["intermediate_steps"]) > max_iterations, # type: ignore
            answer_from_thought_process_runnable | format_last_answer,
        ),
        prompt | llm_with_stop,
    )  # type: ignore
    | ReActSingleInputOutputParser()
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)  # type: ignore
web_crawler_agent_runnable = (
    RunnablePassthrough.assign(
        input={"history": itemgetter("history"), "question": itemgetter("input")}
        | standalone_question_runnable
    )
    | agent_executor
    | itemgetter("output")
)
