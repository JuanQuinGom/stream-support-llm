import os
from typing import List

from langchain.chains.conversation.base import ConversationChain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec
from langchain_ollama import ChatOllama
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain.memory import (
    ConversationBufferWindowMemory,  # Short-term memory
    ConversationSummaryMemory,      # Long-term summarized memory
    CombinedMemory                  # Combines multiple memory types
)
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter
from typing import List


from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.messages import HumanMessage, RemoveMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

INITIAL_PROMPT = """
You are an AI assistant for helping a streamer to strength viewer friendship. 
You have to be friendly or treat the viewer and answer the questions or messages of the user. 
Evaluate also in the message if it is a message or announcement to use a web page for increasing chat viewers or not. 
If so set is_spam to True, otherwise set it to False.
"""


class ViewerAnswer(BaseModel):
    message: str
    is_spam: bool

llm = ChatOllama(
    base_url=os.getenv("LLM_BASE_URL"),
    model="llama2",
    temperature=0.8,
)

# Short-term memory (last 3 messages)
memory = ConversationBufferWindowMemory(
    k=14,
    memory_key="chat_history",
    return_messages=True,  # ← Critical change
    input_key="input"
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(os.getenv("CUSTOM_PROMPT", INITIAL_PROMPT)),
    MessagesPlaceholder(variable_name="chat_history"),  # Recent messages
    ("human", "{input}"),  # New user input
])

# conversation_chain = ConversationChain(
#     llm=llm,
#     memory=memory,
#     prompt=prompt,
#     verbose=True
# )

# structured_llm = llm.with_structured_output(ViewerAnswer)

# async def execute_prompt(prompt: str) -> str:
#     structured_chain = conversation_chain | structured_llm
#
#     response = await structured_chain.ainvoke({
#         "input": prompt
#     })
#     return response


workflow = StateGraph(state_schema=MessagesState)

model = ChatOllama(
    base_url=os.getenv("LLM_BASE_URL"),
    model="llama2",
    temperature=0.8,
    format="json",

)

async def execute_prompt_summary_memory(prompt: str) -> str:

    chat_history = []
    #
    # system_message = SystemMessage(INITIAL_PROMPT)
    #
    # chat_history.append(system_message)
    #
    # chat_history.append(HumanMessage(content="Hello, I'm Neko, how are you?"))
    #
    # response = await model.ainvoke(chat_history)
    #
    # print(response)
    #
    # chat_history.append(HumanMessage(content="What did I say in the previous message?"))
    #
    # response_summary = await model.ainvoke(chat_history)
    # print(response_summary)


    prompt_template = ChatPromptTemplate.from_messages([
        ("system", os.getenv("CUSTOM_PROMPT", INITIAL_PROMPT)),  # System prompt
        MessagesPlaceholder(variable_name="chat_history"),  # Recent messages
        ("human", "{input}"),  # New user input
    ])

    chain = prompt_template | llm

    response = await chain.ainvoke({
        "input": "What is 5+ 2?",
        "chat_history": chat_history
    })

    print(response)

    response_summary = await chain.ainvoke({
        "input": "Now multiply it to 7",
        "chat_history": chat_history
    })

    print(response_summary)


    response_summary_two = await chain.ainvoke({
        "input": "And divide it by 3",
        "chat_history": chat_history
    })

    print(response_summary_two)