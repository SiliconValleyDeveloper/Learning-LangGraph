"""
Phase 1 · Lesson 3 — Watch state change at every step

What you will learn
-------------------
1. stream_mode="values"  — full state after each node
2. stream_mode="updates" — only what that node returned
3. Why reducers matter when multiple nodes write the same key

NO API key needed.

Run:
    python 01_hello_graph/03_watch_state.py
"""

import sys
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from visualize import show_graph


def append_log(existing: list[str], new: list[str]) -> list[str]:
    """Custom reducer: always append, never overwrite."""
    return existing + new


class TraceState(TypedDict):
    counter: int
    log: Annotated[list[str], append_log]


def step_a(state: TraceState) -> dict:
    print("  → running step_a")
    return {"counter": state["counter"] + 1, "log": ["A did +1"]}


def step_b(state: TraceState) -> dict:
    print("  → running step_b")
    return {"counter": state["counter"] + 10, "log": ["B did +10"]}


def step_c(state: TraceState) -> dict:
    print("  → running step_c")
    return {"counter": state["counter"] * 2, "log": ["C did *2"]}


def build_graph():
    builder = StateGraph(TraceState)
    builder.add_node("a", step_a)
    builder.add_node("b", step_b)
    builder.add_node("c", step_c)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", "c")
    builder.add_edge("c", END)
    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    show_graph(graph, title="Watch-state graph")

    initial = {"counter": 0, "log": []}

    print("=== stream_mode='updates' (partial patches) ===\n")
    for chunk in graph.stream(initial, stream_mode="updates"):
        print(f"  {chunk}")

    print("\n=== stream_mode='values' (full state after each node) ===\n")
    for chunk in graph.stream(initial, stream_mode="values"):
        print(f"  counter={chunk['counter']}  log={chunk['log']}")

    print("\nFinal invoke:")
    final = graph.invoke(initial)
    print(f"  {final}")
    print("\nNotice: 'log' grew via the reducer; 'counter' was replaced each time.\n")
