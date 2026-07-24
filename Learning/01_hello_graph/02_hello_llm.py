"""
Phase 1 · Lesson 2 — Chat-style state with a LOCAL LLM (Ollama)

What you will learn
-------------------
1. Messages as state (the usual pattern for chatbots / agents)
2. add_messages reducer — appends instead of overwriting
3. Calling a local LLM inside a node
4. Streaming updates vs one-shot invoke

Needs: Ollama running + a pulled model (default: qwen3:8b)
  ollama pull qwen3:8b

Run:
    python 01_hello_graph/02_hello_llm.py
"""

import sys
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import get_llm, require_ollama
from visualize import show_graph


# ---------------------------------------------------------------------------
# STATE with a REDUCER
# ---------------------------------------------------------------------------
# Without add_messages: new messages would REPLACE old ones.
# With add_messages:    new messages are APPENDED (chat history grows).
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# NODE — call the local LLM with the full message history
# ---------------------------------------------------------------------------
def chatbot_node(state: ChatState) -> dict:
    llm = get_llm(temperature=0.3)

    # System message only for this call; we don't store it permanently here
    # so the graph stays simple. (You'll refine this later.)
    system = SystemMessage(
        content=(
            "You are a friendly tutor teaching LangGraph. "
            "Keep answers under 3 sentences. Do not show chain-of-thought."
        )
    )
    response: AIMessage = llm.invoke([system, *state["messages"]])
    text = response.content if isinstance(response.content, str) else str(response.content)
    print(f"  [chatbot] model replied ({len(text)} chars)")
    return {"messages": [response]}


def build_graph():
    builder = StateGraph(ChatState)
    builder.add_node("chatbot", chatbot_node)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile()


if __name__ == "__main__":
    require_ollama()

    graph = build_graph()
    show_graph(graph, title="Local LLM chatbot graph")

    print("=== Single turn ===\n")
    result = graph.invoke(
        {"messages": [HumanMessage(content="What is a LangGraph node in one sentence?")]}
    )

    for msg in result["messages"]:
        role = msg.__class__.__name__.replace("Message", "")
        print(f"  [{role}] {msg.content}\n")

    print("=== Stream mode: updates (see each node fire) ===\n")
    for update in graph.stream(
        {"messages": [HumanMessage(content="Why do we use state in LangGraph?")]},
        stream_mode="updates",
    ):
        print(f"  update: {list(update.keys())}")

    print("\nNext: Phase 2 — conditional routing (02_router/).\n")
