"""
Phase 8 · Lesson 2 — Durable checkpoints with SQLite

What you will learn
-------------------
1. MemorySaver disappears when the Python process exits
2. SqliteSaver writes checkpoints to a real database file
3. The same thread_id reloads its prior state on the next run
4. Different thread_ids remain isolated

Run twice:
    python Learning/08_advanced/02_sqlite_checkpoint.py

The second run continues the same thread from .data/checkpoints.db.
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from visualize import show_graph

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / ".data" / "checkpoints.db"


class ProfileState(TypedDict):
    message: str
    name: str
    turns: int
    reply: str


def remember_profile(state: ProfileState) -> dict:
    message = state["message"].strip()
    name = state.get("name", "")
    match = re.search(r"\bmy name is\s+([A-Za-z][A-Za-z '-]*)", message, re.IGNORECASE)
    if match:
        name = match.group(1).strip(" .!?")

    turns = state.get("turns", 0) + 1
    if "what is my name" in message.lower():
        reply = f"Your name is {name}." if name else "You have not told me your name yet."
    elif match:
        reply = f"I will remember that your name is {name}."
    else:
        reply = f"Durable turn {turns} recorded."

    return {"name": name, "turns": turns, "reply": reply}


def build_graph(checkpointer: SqliteSaver):
    builder = StateGraph(ProfileState)
    builder.add_node("remember_profile", remember_profile)
    builder.add_edge(START, "remember_profile")
    return builder.compile(checkpointer=checkpointer)


def open_graph(db_path: Path = DEFAULT_DB):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, check_same_thread=False)
    return build_graph(SqliteSaver(connection)), connection


def run_turn(graph, thread_id: str, message: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke({"message": message}, config)


if __name__ == "__main__":
    graph, connection = open_graph()
    try:
        show_graph(graph, title="SQLite-backed profile memory")
        thread = "durable-demo"
        first = run_turn(graph, thread, "My name is Ankit")
        second = run_turn(graph, thread, "What is my name?")
        print(f"\n  first: {first['reply']}")
        print(f"  second: {second['reply']}")
        print(f"  durable turns: {second['turns']}")
        print(f"  database: {DEFAULT_DB}")
    finally:
        connection.close()
