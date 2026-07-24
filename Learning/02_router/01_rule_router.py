"""
Phase 2 · Lesson 1 — Conditional routing (NO LLM yet)

What you will learn
-------------------
1. Conditional edges — pick the next node based on state
2. Router function — returns a string label that matches an edge map
3. Fan-in — multiple branches can end at the same place

Mental model
------------
                    ┌──► weather_handler ──┐
    START → router ─┼──► math_handler    ──┼──► END
                    └──► chitchat_handler ─┘

Run:
    python 02_router/01_rule_router.py
"""

import sys
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from visualize import show_graph


class RouterState(TypedDict):
    user_input: str
    route: str
    answer: str


def router_node(state: RouterState) -> dict:
    """Classify intent with simple rules (no LLM)."""
    text = state["user_input"].lower()
    if any(w in text for w in ("weather", "rain", "temperature", "forecast")):
        route = "weather"
    elif any(w in text for w in ("+", "-", "*", "/", "calculate", "math")):
        route = "math"
    else:
        route = "chitchat"
    print(f"  [router] '{state['user_input']}' → {route}")
    return {"route": route}


def weather_handler(state: RouterState) -> dict:
    return {"answer": "Weather stub: bring an umbrella just in case. (Phase 3: real API)"}


def math_handler(state: RouterState) -> dict:
    # Tiny demo — extracts first "a op b" pattern if present
    text = state["user_input"]
    try:
        # very naive: evaluate only if it looks like a simple expression
        expr = "".join(ch for ch in text if ch in "0123456789+-*/(). ")
        value = eval(expr, {"__builtins__": {}}, {})  # safe: no builtins
        return {"answer": f"Math result: {value}"}
    except Exception:
        return {"answer": "Math stub: try something like 'calculate 12 * 7'"}


def chitchat_handler(state: RouterState) -> dict:
    return {"answer": f"Chitchat: you said '{state['user_input']}'. Nice to meet you!"}


def choose_branch(state: RouterState) -> Literal["weather", "math", "chitchat"]:
    """Used by add_conditional_edges — return value must match the path map keys."""
    return state["route"]  # type: ignore[return-value]


def build_graph():
    builder = StateGraph(RouterState)

    builder.add_node("router", router_node)
    builder.add_node("weather", weather_handler)
    builder.add_node("math", math_handler)
    builder.add_node("chitchat", chitchat_handler)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        choose_branch,
        {
            "weather": "weather",
            "math": "math",
            "chitchat": "chitchat",
        },
    )
    builder.add_edge("weather", END)
    builder.add_edge("math", END)
    builder.add_edge("chitchat", END)

    return builder.compile()


def demo(user_input: str, graph=None) -> None:
    graph = graph or build_graph()
    print(f"\n--- input: {user_input}")
    result = graph.invoke(
        {"user_input": user_input, "route": "", "answer": ""}
    )
    print(f"  route : {result['route']}")
    print(f"  answer: {result['answer']}")


if __name__ == "__main__":
    graph = build_graph()
    show_graph(graph, title="Rule router graph")

    demo("What's the weather in Delhi?", graph)
    demo("calculate 12 * 7", graph)
    demo("Hey, how are you?", graph)
    print(
        "\nTry your own: edit this file or call demo('...') interactively.\n"
        "Next lesson: 02_llm_router.py (LLM picks the route).\n"
    )
