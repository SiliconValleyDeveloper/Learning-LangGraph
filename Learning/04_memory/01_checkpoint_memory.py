"""
Phase 4 · Lesson 1 — Checkpointers + thread_id (multi-turn memory)

What you will learn
-------------------
1. MemorySaver — stores graph state between invokes
2. thread_id — same id = same conversation; different id = fresh chat
3. Why agents without a checkpointer forget everything every call
4. get_state — inspect the latest checkpoint for a thread

Mental model
------------
    invoke(msg, thread="A") ──► checkpoint A
    invoke(msg, thread="A") ──► loads A, appends, saves A again
    invoke(msg, thread="B") ──► separate backpack (no mix)

Needs: Ollama running

Run:
    python 04_memory/01_checkpoint_memory.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import get_llm, require_ollama
from visualize import show_graph


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM = SystemMessage(
    content=(
        "You are a concise assistant. Remember facts the user told you "
        "earlier in THIS conversation. Reply in 1–2 short sentences."
    )
)


def chat_node(state: ChatState) -> dict:
    llm = get_llm(temperature=0)
    response: AIMessage = llm.invoke([SYSTEM, *state["messages"]])
    preview = (response.content or "")[:80]
    print(f"  [chat] {preview!r}")
    return {"messages": [response]}


def build_graph():
    builder = StateGraph(ChatState)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    # Checkpointer is what makes memory persist across invokes
    return builder.compile(checkpointer=MemorySaver())


def say(graph, thread_id: str, text: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n=== thread={thread_id!r} | user: {text} ===")
    result = graph.invoke({"messages": [HumanMessage(content=text)]}, config)
    print(f"  assistant: {result['messages'][-1].content}")


if __name__ == "__main__":
    require_ollama()
    graph = build_graph()
    show_graph(graph, title="Memory chat graph")

    # Same thread → model should remember the name
    say(graph, "user-ankit", "My name is Ankit.")
    say(graph, "user-ankit", "What is my name?")

    snap = graph.get_state({"configurable": {"thread_id": "user-ankit"}})
    print(f"\n  Checkpoint message count: {len(snap.values['messages'])}")
    print(
        "\nNext: 02_two_threads.py — prove two thread_ids never share memory.\n"
    )
