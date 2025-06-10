import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic.v1 import BaseModel

INITIAL_PROMPT = """
You are an AI assistant for helping a streamer to strength viewer friendship. 
You have to be friendly or treat the viewer and answer the questions or messages of the user.
"""

class ViewerAnswer(BaseModel):
    message: str

prompt_template = ChatPromptTemplate.from_messages([
    ("system", os.getenv("CUSTOM PROMPT", INITIAL_PROMPT)),
    ("user", "{user_input}"),
])

llm = ChatOllama(
    base_url=os.getenv("LLM_BASE_URL"),
    model="llama2",
    temperature=0.8,
    format="json",

)
structured_llm = llm.with_structured_output(ViewerAnswer)

async def execute_prompt(prompt: str) -> str:
    structured_chain = prompt_template | structured_llm

    response = await structured_chain.ainvoke({
        "user_input": prompt
    })

    return response.message