"""
Phase 2 · Lesson 2 — LOCAL LLM picks the route (Ollama)

What you will learn
-------------------
1. Using constrained LLM output as a router
2. Same graph shape as 01_rule_router — only the classifier changes
3. Why routing and handling should stay separate nodes

Needs: Ollama running + a pulled model (default: qwen3:8b)

Run:
    python 02_router/02_llm_router.py
"""

import re
import sys
from pathlib import Path
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import get_llm, require_ollama
from visualize import show_graph


class RouterState(TypedDict):
    user_input: str
    route: str
    answer: str


Route = Literal["weather", "math", "chitchat"]


def _extract_route(text: str) -> Route:
    """Parse route label from model output (handles thinking / extra words)."""
    cleaned = text.strip().lower()
    # Drop common thinking blocks some local models emit (e.g. qwen3)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    # Prefer the last non-empty line (final answer after reasoning)
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    candidates = list(reversed(lines)) + [cleaned]
    for chunk in candidates:
        if chunk in ("weather", "math", "chitchat"):
            return chunk  # type: ignore[return-value]
        for label in ("weather", "math", "chitchat"):
            if re.fullmatch(rf".*\b{label}\b.*", chunk) and len(chunk) < 40:
                return label  # type: ignore[return-value]
    for label in ("weather", "math", "chitchat"):
        if re.search(rf"\b{label}\b", cleaned):
            return label  # type: ignore[return-value]
    return "chitchat"


def llm_router_node(state: RouterState) -> dict:
    llm = get_llm(temperature=0)

    prompt = [
        SystemMessage(
            content=(
                "You are an intent classifier.\n"
                "Labels:\n"
                "- weather: rain, forecast, temperature, climate\n"
                "- math: numbers, calculate, arithmetic expressions\n"
                "- chitchat: greetings, opinions, fun facts, anything else\n"
                "Reply with ONLY one word: weather OR math OR chitchat"
            )
        ),
        HumanMessage(content=state["user_input"]),
    ]
    content = llm.invoke(prompt).content
    text = content if isinstance(content, str) else str(content)
    route = _extract_route(text)

    print(f"  [llm_router] '{state['user_input']}' → {route}")
    return {"route": route}


def weather_handler(state: RouterState) -> dict:
    return {"answer": "Weather: (stub) partly cloudy, 28°C. Wire a real API in Phase 3."}


def math_handler(state: RouterState) -> dict:
    text = state["user_input"]
    try:
        expr = "".join(ch for ch in text if ch in "0123456789+-*/(). ")
        value = eval(expr, {"__builtins__": {}}, {})
        return {"answer": f"Math result: {value}"}
    except Exception:
        return {"answer": "Could not parse a simple expression. Try: 'what is 15/3?'"}


def chitchat_handler(state: RouterState) -> dict:
    return {"answer": "Chitchat: I'm your LangGraph router demo. Ask about weather or math!"}


def choose_branch(state: RouterState) -> Route:
    return state["route"]  # type: ignore[return-value]


def build_graph():
    builder = StateGraph(RouterState)
    builder.add_node("router", llm_router_node)
    builder.add_node("weather", weather_handler)
    builder.add_node("math", math_handler)
    builder.add_node("chitchat", chitchat_handler)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        choose_branch,
        {"weather": "weather", "math": "math", "chitchat": "chitchat"},
    )
    builder.add_edge("weather", END)
    builder.add_edge("math", END)
    builder.add_edge("chitchat", END)
    return builder.compile()


def demo(user_input: str, graph=None) -> None:
    graph = graph or build_graph()
    print(f"\n--- input: {user_input}")
    result = graph.invoke({"user_input": user_input, "route": "", "answer": ""})
    print(f"  route : {result['route']}")
    print(f"  answer: {result['answer']}")


if __name__ == "__main__":
    require_ollama()

    graph = build_graph()
    show_graph(graph, title="Local LLM router graph")

    demo("Will it rain tomorrow in Mumbai?", graph)
    demo("What is 144 / 12?", graph)
    demo("Hey, tell me a joke!", graph)
    print("\nPhase 2 done. Next up: Phase 3 — tools + ReAct agent loop.\n")
