from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

from src.config.settings import get_settings


def _get_llm():
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0,
        convert_system_message_to_human=True,
        max_retries=2,
    ) # type: ignore

def get_llm():
    return _get_llm() | StrOutputParser()
