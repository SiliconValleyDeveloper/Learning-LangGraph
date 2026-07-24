"""
Phase 4 · Lesson 2 — Thread isolation (two chats, no mixing)

What you will learn
-------------------
1. Different thread_id → completely separate state
2. Same graph instance can serve many users safely
3. State history — get_state_history for debugging / replay

Needs: Ollama running

Run:
    python 04_memory/02_two_threads.py
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


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM = SystemMessage(
    content=(
        "You are a concise assistant. Use only facts from THIS thread. "
        "If you do not know, say you do not know. 1 sentence max."
    )
)


def chat_node(state: ChatState) -> dict:
    llm = get_llm(temperature=0)
    response: AIMessage = llm.invoke([SYSTEM, *state["messages"]])
    return {"messages": [response]}


def build_graph():
    builder = StateGraph(ChatState)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    return builder.compile(checkpointer=MemorySaver())


def say(graph, thread_id: str, text: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(content=text)]}, config)
    reply = result["messages"][-1].content
    print(f"  [{thread_id}] user={text!r}")
    print(f"  [{thread_id}] bot ={reply!r}\n")
    return reply if isinstance(reply, str) else str(reply)


if __name__ == "__main__":
    require_ollama()
    graph = build_graph()

    print("=== Seed two separate threads ===\n")
    say(graph, "alice", "Remember: my favorite color is blue.")
    say(graph, "bob", "Remember: my favorite color is green.")

    print("=== Ask each thread — answers must NOT cross ===\n")
    say(graph, "alice", "What is my favorite color?")
    say(graph, "bob", "What is my favorite color?")

    # History: how many checkpoints were written for alice?
    history = list(graph.get_state_history({"configurable": {"thread_id": "alice"}}))
    print(f"  alice checkpoints in history: {len(history)}")
    print(
        "\nPhase 4 done. Next: Phase 5 — human-in-the-loop "
        "(pause, approve, edit, resume).\n"
    )
