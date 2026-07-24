"""
Phase 7 · Lesson 1 — Limits, tool errors, and safe recovery

What you will learn
-------------------
1. recursion_limit — hard cap on graph steps (stops runaway loops)
2. Tools that fail — return an error STRING instead of crashing the graph
3. Agent reads the error ToolMessage and recovers (or explains failure)
4. Why crashing inside a node is worse than returning a soft error

Needs: Ollama running

Run:
    python 07_production/01_limits_errors.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import get_llm, require_ollama
from visualize import show_graph


@tool
def safe_divide(a: float, b: float) -> str:
    """Divide a by b. Returns an error string if b is zero (does not crash)."""
    if b == 0:
        return "Error: division by zero — ask the user for a non-zero divisor."
    return str(a / b)


@tool
def flaky_lookup(key: str) -> str:
    """Look up a fake catalog key. Unknown keys return a soft error."""
    catalog = {"langgraph": "A library for stateful agent graphs.", "ollama": "Local LLM runner."}
    if key.lower() not in catalog:
        return f"Error: unknown key '{key}'. Known: {', '.join(catalog)}"
    return catalog[key.lower()]


TOOLS = [safe_divide, flaky_lookup]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM = SystemMessage(
    content=(
        "You have tools: safe_divide, flaky_lookup. "
        "If a tool returns Error: ..., explain it briefly and suggest a fix. "
        "Do not invent tool results."
    )
)


def agent_node(state: AgentState) -> dict:
    llm = get_llm(temperature=0).bind_tools(TOOLS)
    response: AIMessage = llm.invoke([SYSTEM, *state["messages"]])
    print(f"  [agent] tool_calls={len(response.tool_calls or [])}")
    return {"messages": [response]}


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile()


def run_query(graph, text: str) -> None:
    print(f"\n=== User: {text} ===\n")
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=text)]},
            config={"recursion_limit": 8},  # safety cap
        )
        print(f"  Final: {result['messages'][-1].content}\n")
    except Exception as exc:  # noqa: BLE001 — show limit / other failures clearly
        print(f"  Graph stopped: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    require_ollama()
    graph = build_graph()
    show_graph(graph, title="Production-safe tool agent")

    run_query(graph, "What is 10 divided by 0?")
    run_query(graph, "Lookup key: banana")
    run_query(graph, "Lookup key: langgraph")

    print(
        "Next: 02_stream_modes.py — stream updates for a better UX.\n"
    )
