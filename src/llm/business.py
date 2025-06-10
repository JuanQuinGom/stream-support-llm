import os

from langchain_ollama import ChatOllama
from pydantic.v1 import BaseModel

INITIAL_PROMPT = """
You are an AI assistant for helping a streamer to strength viewer friendship. 
You have to be friendly or treat the viewer and answer the questions or messages of the user.
"""

class ViewerAnswer(BaseModel):
    message: str

llm = ChatOllama(
    base_url=os.getenv("LLM_BASE_URL"),
    model="llama2",
    temperature=0.8,
    format="json",
)
structured_llm = llm.with_structured_output(ViewerAnswer)

async def execute_prompt(prompt: str) -> str:
    custom_prompt = os.getenv("CUSTOM PROMPT", None)
    system_message = custom_prompt if custom_prompt else INITIAL_PROMPT
    messages = [
        ("system", system_message),
        ("message", prompt),
    ]

    response = await structured_llm.ainvoke(messages)

    return response.message