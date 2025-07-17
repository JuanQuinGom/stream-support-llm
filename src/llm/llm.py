import os
from typing import Any, TypedDict

from langchain_core.messages import AnyMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import SummarizationNode

model = ChatOllama(
    base_url=os.getenv("LLM_BASE_URL"),
    model="llama2",
    temperature=0.8,
)

summarization_model = model

class State(MessagesState):
    context: dict[str, Any]


class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, Any]


def init_llm_model():
    summarization_node = SummarizationNode(
        token_counter=count_tokens_approximately,
        model=summarization_model,
        max_tokens=1024,
        max_tokens_before_summary=256,
        max_summary_tokens=128,
    )

    def call_model(state: LLMInputState):
        response = model.invoke(state["summarized_messages"])
        return {"messages": [response]}

    checkpointer = InMemorySaver()
    builder = StateGraph(State)
    builder.add_node(call_model)
    builder.add_node("summarize", summarization_node)
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "call_model")
    builder.add_edge("call_model", END)
    graph = builder.compile(checkpointer=checkpointer)
    return graph

def execute_graph_prompt(thread_id: int, graph: StateGraph, message: str):
    config = {"configurable": {"thread_id": str(thread_id)}}
    response = graph.invoke({"messages": [{"content": message, "role": "user"}]}, config)  # Fix message format

    return response["messages"][-1]

# if __name__ == "__main__":
#     graph = init_llm_model()
#     print("Graph initialized successfully.")
#
#     # Example usage of the graph
#     execute_graph_prompt(1, graph, "Hi, my name is Bob")
#     execute_graph_prompt(1, graph, "Write a short poem about cats")
#     execute_graph_prompt(1, graph, "Now do the same but for dogs")
#
#     # Get summarize of a question
#     execute_graph_prompt(1, graph, "What's my name?")
#     execute_graph_prompt(2, graph, "What's my name?")
