"""
Phase 6 · Lesson 1 — Supervisor routes to specialist agents

What you will learn
-------------------
1. Multi-agent = multiple specialized nodes + a supervisor router
2. Supervisor decides who works next based on the task type
3. Specialists return results into shared state; supervisor may stop
4. Same conditional-edge idea as Phase 2 — now with "workers"

Mental model
------------
                    ┌──► researcher ──┐
    START → supervisor ─┼──► writer     ─┼──► (back to supervisor) → END
                    └──► END (done)   ─┘

Needs: Ollama running (supervisor classification)

Run:
    python 06_multi_agent/01_supervisor.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import get_llm, require_ollama
from visualize import show_graph


class TeamState(TypedDict):
    task: str
    research_notes: str
    draft: str
    next_worker: str
    steps: int


Worker = Literal["researcher", "writer", "done"]


def _extract_worker(text: str) -> Worker:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip().lower()
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    for chunk in list(reversed(lines)) + [cleaned]:
        for label in ("researcher", "writer", "done"):
            if chunk == label or re.search(rf"\b{label}\b", chunk):
                return label  # type: ignore[return-value]
    return "done"


def supervisor_node(state: TeamState) -> dict:
    """Decide who should act next (or done)."""
    llm = get_llm(temperature=0)
    prompt = [
        SystemMessage(
            content=(
                "You are a supervisor for a tiny writing team.\n"
                "Workers:\n"
                "- researcher: gather bullet facts about a topic\n"
                "- writer: turn research_notes into a short paragraph\n"
                "- done: finish when draft is good enough OR steps >= 4\n"
                "Rules:\n"
                "- If research_notes is empty → researcher\n"
                "- If notes exist but draft is empty → writer\n"
                "- If draft exists → done\n"
                "Reply with ONLY one word: researcher OR writer OR done"
            )
        ),
        HumanMessage(
            content=(
                f"task: {state['task']}\n"
                f"research_notes: {state['research_notes']!r}\n"
                f"draft: {state['draft']!r}\n"
                f"steps: {state['steps']}"
            )
        ),
    ]
    content = llm.invoke(prompt).content
    text = content if isinstance(content, str) else str(content)
    nxt = _extract_worker(text)
    if state["steps"] >= 4:
        nxt = "done"
    print(f"  [supervisor] next={nxt} (step {state['steps']})")
    return {"next_worker": nxt, "steps": state["steps"] + 1}


def researcher_node(state: TeamState) -> dict:
    # Stub research — swap for a real search tool later
    topic = state["task"]
    notes = (
        f"- Topic: {topic}\n"
        "- LangGraph uses stateful graphs (nodes + edges).\n"
        "- Checkpointers give multi-turn memory via thread_id.\n"
        "- HITL pauses graphs before risky actions."
    )
    print("  [researcher] filled notes")
    return {"research_notes": notes}


def writer_node(state: TeamState) -> dict:
    notes = state["research_notes"] or "(no notes)"
    draft = (
        f"Summary for '{state['task']}':\n"
        f"Based on research:\n{notes}\n"
        "In short: LangGraph lets you build reliable agent workflows "
        "with memory and human approval."
    )
    print("  [writer] wrote draft")
    return {"draft": draft}


def route_from_supervisor(state: TeamState) -> Worker:
    return state["next_worker"]  # type: ignore[return-value]


def build_graph():
    builder = StateGraph(TeamState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            "writer": "writer",
            "done": END,
        },
    )
    # Workers always report back to the supervisor
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("writer", "supervisor")
    return builder.compile()


if __name__ == "__main__":
    require_ollama()
    graph = build_graph()
    show_graph(graph, title="Supervisor multi-agent graph")

    result = graph.invoke(
        {
            "task": "Explain LangGraph memory + HITL in simple terms",
            "research_notes": "",
            "draft": "",
            "next_worker": "",
            "steps": 0,
        },
        config={"recursion_limit": 12},
    )

    print("\n=== Final draft ===\n")
    print(result["draft"])
    print(
        "\nNext: 02_parallel_fanout.py — run several workers at once, then merge.\n"
    )
