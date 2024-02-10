from typing import List, Tuple
from uuid import uuid4

import gradio as gr
from langchain.schema import HumanMessage, AIMessage

from src.runnables.omnibot_agent import omnibot_runnable
from src.runnables.web_crawler import web_crawler_runnable
from src.constants import OMNIBOT_DESCRIPTION, CHAT_DESCRIPTION, CRAWLER_DESCRIPTION
from src.logging import setup_logging


def respond(message: str, chat_history: List[Tuple[str, str]]):
    history = []
    for human_message, bot_message in chat_history:
        history.append(HumanMessage(content=human_message))
        history.append(AIMessage(content=bot_message))
    bot_message = omnibot_runnable.invoke(
        {"input": message, "history": history}
    )
    chat_history.append((message, bot_message))
    return "", chat_history


def answer_question(question: str):
    return web_crawler_runnable.invoke({"question": question})


if __name__ == "__main__":
    setup_logging()

    with gr.Blocks(theme=gr.themes.Base(), title="OmniBot") as demo:
        gr.Markdown("# OmniBot")
        
        with gr.Row() as row:
            with gr.Column() as left:
                gr.Image("./assets/omnibot.jpeg")
            with gr.Column(scale=2) as right:
                gr.Markdown(OMNIBOT_DESCRIPTION)
        with gr.Tab(label="Chat") as chat:
            gr.Markdown(CHAT_DESCRIPTION)
            chatbot = gr.Chatbot()
            msg = gr.Textbox()
            clear = gr.ClearButton([msg, chatbot])
            msg.submit(respond, [msg, chatbot], [msg, chatbot])
        with gr.Tab(label="Crawler") as crawler:
            gr.Markdown(CRAWLER_DESCRIPTION)
            question = gr.Textbox(label="Question")
            search = gr.Button(value="Search")
            output = gr.Markdown()
            search.click(answer_question, inputs=[question], outputs=[output])
    demo.launch(share=True)
