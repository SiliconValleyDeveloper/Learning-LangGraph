"""
Phase 5 · Lesson 2 — Edit state mid-run, then resume

What you will learn
-------------------
1. update_state(config, patch) — human edits the checkpoint
2. Resume after edit — next node sees the NEW values
3. Reject path — update status and jump / end without sending

NO Ollama needed.

Run:
    python 05_hitl/02_edit_and_resume.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from visualize import show_graph


class EmailState(TypedDict):
    to: str
    subject: str
    body: str
    status: str


def draft_node(state: EmailState) -> dict:
    body = f"Draft for {state['to']}: {state['body']}"
    print(f"  [draft] {body!r}")
    return {"body": body, "status": "awaiting_approval"}


def send_node(state: EmailState) -> dict:
    print(f"  [send]  SENDING → {state['to']}: {state['body']!r}")
    return {"status": "sent"}


def build_graph():
    builder = StateGraph(EmailState)
    builder.add_node("draft", draft_node)
    builder.add_node("send", send_node)
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "send")
    builder.add_edge("send", END)
    return builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["send"],
    )


if __name__ == "__main__":
    graph = build_graph()
    show_graph(graph, title="Edit-then-resume graph")

    config = {"configurable": {"thread_id": "email-edit-1"}}

    print("=== 1) Draft + pause ===\n")
    graph.invoke(
        {
            "to": "boss@example.com",
            "subject": "PTO",
            "body": "I need Friday off.",
            "status": "",
        },
        config,
    )
    print(f"  paused body: {graph.get_state(config).values['body']!r}")

    print("\n=== 2) Human edits the draft ===\n")
    graph.update_state(
        config,
        {
            "body": "Draft for boss@example.com: I need Friday AND Monday off.",
            "status": "edited_by_human",
        },
    )
    print(f"  edited body: {graph.get_state(config).values['body']!r}")

    print("\n=== 3) Resume → send uses edited body ===\n")
    final = graph.invoke(None, config)
    print(f"\n  Final status: {final['status']}")
    print(
        "\nPhase 5 done. Next: Phase 6 — supervisor + multi-agent patterns.\n"
    )
