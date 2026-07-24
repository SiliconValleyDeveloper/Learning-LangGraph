"""
Phase 3 · Lesson 2 — ReAct agent loop (agent ↔ tools)

What you will learn
-------------------
1. ToolNode — executes tool_calls from the last AIMessage
2. tools_condition — route to tools if tool_calls exist, else END
3. The loop: agent → (tools → agent)* → END
4. add_messages — chat + ToolMessages accumulate in state

Mental model
------------
                    ┌──────────────┐
                    │              │ tool_calls?
    START ──► agent ┤              ▼
                    │           tools
                    │              │
                    │◄─────────────┘  (ToolMessage back into state)
                    │
                    └─ no tool_calls ──► END

Needs: Ollama running

Run:
    python 03_tools_agent/02_react_agent.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import get_llm, require_ollama
from visualize import show_graph


# ---------------------------------------------------------------------------
# Tools (fake / local — no external APIs)
# ---------------------------------------------------------------------------
@tool
def calculator(expression: str) -> str:
    """Evaluate a simple math expression like '12 * 7' or '(100-8)/4'."""
    allowed = set("0123456789+-*/(). %")
    if not expression or any(ch not in allowed for ch in expression):
        return "Error: only digits and + - * / ( ) . % are allowed"
    try:
        value = eval(expression, {"__builtins__": {}}, {})
        return str(value)
    except Exception as exc:  # noqa: BLE001 — surface to the agent
        return f"Error: {exc}"


@tool
def get_time(city: str) -> str:
    """Return the current local time for a known city (Delhi, London, New_York, Tokyo)."""
    zones = {
        "delhi": "Asia/Kolkata",
        "mumbai": "Asia/Kolkata",
        "london": "Europe/London",
        "new_york": "America/New_York",
        "tokyo": "Asia/Tokyo",
    }
    key = city.strip().lower().replace(" ", "_")
    tz_name = zones.get(key)
    if not tz_name:
        return f"Unknown city '{city}'. Try: Delhi, London, New_York, Tokyo."
    now = datetime.now(ZoneInfo(tz_name))
    return now.strftime(f"%Y-%m-%d %H:%M:%S ({tz_name})")


@tool
def word_count(text: str) -> int:
    """Count whitespace-separated words in the given text."""
    return len(text.split())


TOOLS = [calculator, get_time, word_count]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
SYSTEM = SystemMessage(
    content=(
        "You are a helpful assistant with tools: calculator, get_time, word_count. "
        "Use tools when they help. After tools return, give a short final answer. "
        "Do not invent tool results."
    )
)


def agent_node(state: AgentState) -> dict:
    llm = get_llm(temperature=0).bind_tools(TOOLS)
    response: AIMessage = llm.invoke([SYSTEM, *state["messages"]])
    n_calls = len(response.tool_calls or [])
    preview = (response.content or "")[:80]
    print(f"  [agent] tool_calls={n_calls}  content={preview!r}")
    return {"messages": [response]}


def build_graph(checkpointer=None):
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "agent")
    # If the last AI message has tool_calls → "tools", else → END
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")  # loop back with ToolMessages

    return builder.compile(checkpointer=checkpointer)


def run_query(graph, text: str) -> None:
    print(f"\n=== User: {text} ===\n")
    result = graph.invoke({"messages": [HumanMessage(content=text)]})
    final = result["messages"][-1]
    print(f"\n  Final: {final.content}\n")


if __name__ == "__main__":
    require_ollama()
    graph = build_graph()
    show_graph(graph, title="ReAct agent graph")

    run_query(graph, "What is 144 / 12?")
    run_query(graph, "What time is it in Delhi right now?")
    run_query(graph, "How many words are in: LangGraph makes agent loops clear?")

    print(
        "Try your own query: edit run_query(...) above.\n"
        "Next: Phase 4 — memory + checkpointers "
        "(python 04_memory/01_checkpoint_memory.py).\n"
    )
