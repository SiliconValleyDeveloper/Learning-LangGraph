"""
Phase 1 · Lesson 1 — Hello LangGraph (NO API key needed)

What you will learn
-------------------
1. STATE  — shared data the graph carries (like a backpack)
2. NODES  — Python functions that read state and return updates
3. EDGES  — connections that decide the next node
4. COMPILE + INVOKE — turn the blueprint into something you can run
5. VISUALIZE — draw_ascii() + draw_mermaid() (built into LangGraph)

Mental model
------------
    START ──► greet ──► echo ──► END
              │          │
              └─ update  └─ update
                 state      state

Run:
    python 01_hello_graph/01_hello_state.py
"""

import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

# Allow: python 01_hello_graph/01_hello_state.py  (project root on path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from visualize import show_graph


# ---------------------------------------------------------------------------
# 1) Define STATE — the shape of data that flows through the graph
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    name: str
    greeting: str
    echo: str
    step_log: list[str]  # we append here so you can see the journey


# ---------------------------------------------------------------------------
# 2) Define NODES — each receives full state, returns a PARTIAL update
# ---------------------------------------------------------------------------
def greet_node(state: GraphState) -> dict:
    """Create a greeting from the user's name."""
    name = state["name"]
    greeting = f"Hello, {name}! Welcome to LangGraph."
    print(f"  [greet] created greeting for '{name}'")
    return {
        "greeting": greeting,
        "step_log": state["step_log"] + ["greet"],
    }


def echo_node(state: GraphState) -> dict:
    """Repeat the greeting in uppercase (shows reading prior node output)."""
    greeting = state["greeting"]
    echo = greeting.upper()
    print(f"  [echo]  echoed: {echo}")
    return {
        "echo": echo,
        "step_log": state["step_log"] + ["echo"],
    }


# ---------------------------------------------------------------------------
# 3) Wire the GRAPH — nodes + edges
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("greet", greet_node)
    builder.add_node("echo", echo_node)

    # START → greet → echo → END
    builder.add_edge(START, "greet")
    builder.add_edge("greet", "echo")
    builder.add_edge("echo", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# 4) Run it
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    graph = build_graph()

    # Built-in visualization (no Cursor plugin / extension needed)
    show_graph(graph, title="Hello graph")

    print("=== Invoking graph ===\n")
    result = graph.invoke(
        {
            "name": "Ankit",
            "greeting": "",
            "echo": "",
            "step_log": [],
        }
    )

    print("\n=== Final state ===")
    print(f"  name     : {result['name']}")
    print(f"  greeting : {result['greeting']}")
    print(f"  echo     : {result['echo']}")
    print(f"  step_log : {result['step_log']}")
    print("\nTry: change 'Ankit' above, or swap node order (echo before greet).\n")
