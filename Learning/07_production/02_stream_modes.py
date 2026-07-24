"""
Phase 7 · Lesson 2 — Streaming modes (values / updates)

What you will learn
-------------------
1. stream_mode="updates" — only what each node changed (great for logs/UI)
2. stream_mode="values" — full state after every step
3. Why APIs/UIs prefer streaming over waiting for the final invoke()
4. Capstone hint — combine tools + memory + HITL + stream in one app

NO Ollama needed (pure state graph).

Run:
    python 07_production/02_stream_modes.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from visualize import show_graph


def append_log(existing: list[str], new: list[str]) -> list[str]:
    return existing + new


class JobState(TypedDict):
    item: str
    score: int
    log: Annotated[list[str], append_log]


def normalize(state: JobState) -> dict:
    item = state["item"].strip().title()
    print(f"  [normalize] {item!r}")
    return {"item": item, "log": ["normalized"]}


def score_node(state: JobState) -> dict:
    score = len(state["item"])
    print(f"  [score] {score}")
    return {"score": score, "log": [f"scored={score}"]}


def label_node(state: JobState) -> dict:
    label = "short" if state["score"] < 8 else "long"
    print(f"  [label] {label}")
    return {"log": [f"label={label}"]}


def build_graph():
    builder = StateGraph(JobState)
    builder.add_node("normalize", normalize)
    builder.add_node("score", score_node)
    builder.add_node("label", label_node)
    builder.add_edge(START, "normalize")
    builder.add_edge("normalize", "score")
    builder.add_edge("score", "label")
    builder.add_edge("label", END)
    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    show_graph(graph, title="Streaming demo graph")

    seed = {"item": "  langgraph  ", "score": 0, "log": []}

    print("=== stream_mode='updates' (per-node patches) ===\n")
    for chunk in graph.stream(seed, stream_mode="updates"):
        print(f"  update: {chunk}")

    print("\n=== stream_mode='values' (full state each step) ===\n")
    for value in graph.stream(seed, stream_mode="values"):
        print(f"  values: item={value['item']!r} score={value['score']} log={value['log']}")

    print(
        "\nPhase 7 done. Capstone idea:\n"
        "  ReAct agent (Phase 3) + MemorySaver (4) + interrupt_before (5)\n"
        "  + stream updates to your Angular/FastAPI UI.\n"
        "You now have the full basic → advanced LangGraph path.\n"
    )
