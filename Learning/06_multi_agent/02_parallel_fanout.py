"""
Phase 6 · Lesson 2 — Parallel fan-out + merge (map-reduce style)

What you will learn
-------------------
1. Send — dispatch the SAME node many times with different payloads
2. Fan-out — workers run as separate branches
3. Reducer — Annotated[list, operator.add] merges worker results
4. Fan-in — after all Sends finish, the merge node runs once

Mental model
------------
                    ┌─► worker(topic=A) ─┐
    START → plan ───┼─► worker(topic=B) ─┼─► merge → END
                    └─► worker(topic=C) ─┘

NO Ollama needed.

Run:
    python 06_multi_agent/02_parallel_fanout.py
"""

from __future__ import annotations

import operator
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from visualize import show_graph


class FanoutState(TypedDict):
    topics: list[str]
    # Each worker returns a list of 1 note; reducer concatenates them
    notes: Annotated[list[str], operator.add]
    summary: str


def plan_node(state: FanoutState) -> dict:
    print(f"  [plan] topics={state['topics']}")
    return {}


def worker_node(state: FanoutState) -> dict:
    """Runs once per Send — state here is a partial payload."""
    topic = state["topics"][0]  # each Send passes a 1-item topics list
    note = f"Note about {topic}: practiced in LangGraph phase 6."
    print(f"  [worker] {topic!r}")
    return {"notes": [note]}


def merge_node(state: FanoutState) -> dict:
    summary = " | ".join(state["notes"])
    print(f"  [merge] combined {len(state['notes'])} notes")
    return {"summary": summary}


def fan_out(state: FanoutState) -> list[Send]:
    """Return one Send per topic → parallel worker invocations."""
    return [Send("worker", {"topics": [t], "notes": [], "summary": ""}) for t in state["topics"]]


def build_graph():
    builder = StateGraph(FanoutState)
    builder.add_node("plan", plan_node)
    builder.add_node("worker", worker_node)
    builder.add_node("merge", merge_node)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", fan_out, ["worker"])
    builder.add_edge("worker", "merge")
    builder.add_edge("merge", END)
    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    show_graph(graph, title="Fan-out / map-reduce graph")

    result = graph.invoke(
        {
            "topics": ["checkpointers", "HITL", "supervisors"],
            "notes": [],
            "summary": "",
        }
    )

    print("\n=== Notes ===")
    for n in result["notes"]:
        print(f"  - {n}")
    print(f"\n=== Summary ===\n  {result['summary']}")
    print(
        "\nPhase 6 done. Next: Phase 7 — limits, errors, streaming "
        "(production mindset).\n"
    )
