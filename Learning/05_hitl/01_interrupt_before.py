"""
Phase 5 · Lesson 1 — interrupt_before (pause for human approval)

What you will learn
-------------------
1. interrupt_before=["node"] — graph stops BEFORE that node runs
2. Checkpointer is required for interrupts (state must be saved)
3. Resume with invoke(None, config) — continues from the pause
4. get_state(...).next — see which node is waiting

Mental model
------------
    START → draft → ⏸ (interrupt before send) → send → END
                         │
                    human says "ok"
                         │
                    invoke(None) resumes

NO Ollama needed for this lesson.

Run:
    python 05_hitl/01_interrupt_before.py
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
    body = (
        f"Hi {state['to'].split('@')[0].title()},\n\n"
        f"{state['body']}\n\n"
        "Best regards,\nLangGraph Demo"
    )
    print("  [draft] wrote email body")
    return {"body": body, "status": "drafted"}


def send_node(state: EmailState) -> dict:
    # Mock send — in real life this hits an email API
    print(f"  [send]  To: {state['to']}")
    print(f"  [send]  Subject: {state['subject']}")
    print(f"  [send]  Body:\n{state['body']}")
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
        interrupt_before=["send"],  # ← pause here for approval
    )


if __name__ == "__main__":
    graph = build_graph()
    show_graph(graph, title="HITL email graph")

    config = {"configurable": {"thread_id": "email-1"}}
    initial = {
        "to": "alice@example.com",
        "subject": "Meeting tomorrow",
        "body": "Can we meet at 3pm?",
        "status": "",
    }

    print("=== Run until interrupt ===\n")
    paused = graph.invoke(initial, config)
    snap = graph.get_state(config)
    print(f"  status after pause : {paused['status']}")
    print(f"  next node waiting  : {snap.next}")

    print("\n=== Human approves → resume ===\n")
    final = graph.invoke(None, config)  # None = continue from checkpoint
    print(f"\n  Final status: {final['status']}")
    print(
        "\nNext: 02_edit_and_resume.py — change the draft, then continue.\n"
    )
