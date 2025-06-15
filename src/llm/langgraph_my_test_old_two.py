import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, StateGraph, START
from langmem.short_term import SummarizationNode
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.checkpoint.memory import InMemorySaver
from typing import Any

load_dotenv()

model  = ChatOllama(
    base_url=os.getenv("LLM_BASE_URL"),
    model="llama2",
    temperature=0.8,
    format="json",

)

summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=model,
    max_tokens=384,
    max_summary_tokens=128,
    output_messages_key="llm_input_messages",
)

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


class State(AgentState):
    # NOTE: we're adding this key to keep track of previous summary information
    # to make sure we're not summarizing on every LLM call
    context: dict[str, Any]

checkpointer = InMemorySaver()

agent = create_react_agent(
    model=model,
    tools=[],
    pre_model_hook=summarization_node,
    state_schema=State,
    checkpointer=checkpointer,
)

# Run the agent
config = {
    "configurable": {
        "thread_id": "1"


    }
}

sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    config
)
print(sf_response)
config = {
    "configurable": {
        "thread_id": "1"


    }
}
# Continue the conversation using the same thread_id
ny_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what about new york?"}]},
    config
)

print(ny_response)

# This is storing the state in memory, so we can continue the conversation in the same thread

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

builder = StateGraph(MessagesState)
builder.add_node(call_model)
builder.add_edge(START, "call_model")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {
    "configurable": {
        "thread_id": "1"
    }
}

for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
        config,
        stream_mode="values",
):
    chunk["messages"][-1].pretty_print()

for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "what's my name?"}]},
        config,
        stream_mode="values",
):
    chunk["messages"][-1].pretty_print()



import uuid
from typing_extensions import Annotated, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

def call_model_long_term(
        state: MessagesState,
        config: RunnableConfig,
        *,
        store: BaseStore,


):
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)
    memories = store.search(namespace, query=str(state["messages"][-1].content))
    info = "\n".join([d.value["data"] for d in memories])
    system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

    # Store new memories if the user asks the model to remember
    last_message = state["messages"][-1]
    if "remember" in last_message.content.lower():
        memory = "User name is Bob"
        store.put(namespace, str(uuid.uuid4()), {"data": memory})

    response = model.invoke(
        [{"role": "system", "content": system_msg}] + state["messages"]
    )
    return {"messages": response}


builder = StateGraph(MessagesState)
builder.add_node(call_model_long_term)
builder.add_edge(START, "call_model_long_term")

checkpointer = InMemorySaver()
store = InMemoryStore()

graph = builder.compile(
    checkpointer=checkpointer,
    store=store,
)


config = {
    "configurable": {
        "thread_id": "1",
        "user_id": "1",
    }
}
for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "Hi! Remember: my name is Bob"}]},
        config,
        stream_mode="values",
):
    chunk["messages"][-1].pretty_print()

config = {
    "configurable": {
        "thread_id": "2",
        "user_id": "1",
    }
}

for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "what is my name?"}]},
        config,
        stream_mode="values",
):
    chunk["messages"][-1].pretty_print()
