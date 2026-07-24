"""
Phase 8 · Lesson 1 — Compose a graph from a reusable subgraph

What you will learn
-------------------
1. A compiled graph can be used as a node inside another graph
2. Parent and child graphs can share a state schema
3. Subgraphs isolate a reusable workflow without hiding its state
4. Parent orchestration stays small while the child owns its internal steps

Run:
    python Learning/08_advanced/01_subgraph.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from visualize import show_graph


class WritingState(TypedDict):
    topic: str
    outline: str
    draft: str
    review: str
    log: list[str]


def make_outline(state: WritingState) -> dict:
    outline = f"Define {state['topic']} → show composition → explain reuse"
    return {"outline": outline, "log": [*state["log"], "child:outline"]}


def write_draft(state: WritingState) -> dict:
    draft = (
        f"{state['topic']} becomes easier to maintain when a focused workflow "
        f"is compiled once and reused. Outline: {state['outline']}."
    )
    return {"draft": draft, "log": [*state["log"], "child:draft"]}


def build_writing_subgraph():
    child = StateGraph(WritingState)
    child.add_node("make_outline", make_outline)
    child.add_node("write_draft", write_draft)
    child.add_edge(START, "make_outline")
    child.add_edge("make_outline", "write_draft")
    child.add_edge("write_draft", END)
    return child.compile()


def prepare(state: WritingState) -> dict:
    return {
        "topic": state["topic"].strip().title(),
        "log": [*state["log"], "parent:prepare"],
    }


def review(state: WritingState) -> dict:
    word_count = len(state["draft"].split())
    return {
        "review": f"Approved: focused draft with {word_count} words.",
        "log": [*state["log"], "parent:review"],
    }


def build_graph():
    parent = StateGraph(WritingState)
    parent.add_node("prepare", prepare)
    parent.add_node("writing_subgraph", build_writing_subgraph())
    parent.add_node("review", review)
    parent.add_edge(START, "prepare")
    parent.add_edge("prepare", "writing_subgraph")
    parent.add_edge("writing_subgraph", "review")
    parent.add_edge("review", END)
    return parent.compile()


def initial_state(topic: str) -> WritingState:
    return {
        "topic": topic,
        "outline": "",
        "draft": "",
        "review": "",
        "log": [],
    }


if __name__ == "__main__":
    graph = build_graph()
    show_graph(graph, title="Parent graph with a writing subgraph")

    result = graph.invoke(initial_state("reusable LangGraph subgraphs"))
    print("\n=== Result ===")
    print(f"  draft: {result['draft']}")
    print(f"  review: {result['review']}")
    print(f"  path: {' → '.join(result['log'])}")
    print("\nNext: 02_sqlite_checkpoint.py — persist state across processes.\n")
