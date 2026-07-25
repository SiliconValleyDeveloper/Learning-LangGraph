"""
Phase 13 · Lesson 1 — MCP tools inside a LangGraph ReAct agent

What you will learn
-------------------
1. MCP server exposes tools over a standard protocol (stdio here)
2. langchain-mcp-adapters converts MCP tools → LangChain tools
3. Same ReAct loop as Phase 3: agent ↔ ToolNode until the LLM answers
4. Why MCP: share the same tools with Cursor, other agents, other apps

Mental model
------------
    demo_mcp_server.py (MCP)
            │ stdio
            ▼
    MultiServerMCPClient  →  LangChain tools
            │
            ▼
    START → agent ⇄ tools → END     (Ollama LLM)

Needs: Ollama running +  pip install langchain-mcp-adapters mcp

Run (from repo root):
    python Learning/13_mcp_langgraph/01_mcp_tools_agent.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import get_llm, require_ollama
from visualize import show_graph

HERE = Path(__file__).resolve().parent
SERVER = HERE / "demo_mcp_server.py"


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM = SystemMessage(
    content=(
        "You are a market helper with MCP tools: list_tickers, get_quote, fx_rate. "
        "Always call tools for prices/rates — do not invent numbers. "
        "These quotes are DEMO data from a local MCP server. "
        "Keep the final answer under 4 sentences."
    )
)


def _stdio_connection() -> dict:
    """Spawn the demo MCP server the same way Cursor spawns MCP via stdio."""
    return {
        "demo_market": {
            "command": sys.executable,
            "args": [str(SERVER)],
            "transport": "stdio",
        }
    }


async def load_mcp_tools():
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(_stdio_connection())
    tools = await client.get_tools()
    names = [t.name for t in tools]
    print(f"  [mcp] loaded {len(tools)} tool(s): {', '.join(names)}")
    return tools


def build_graph(tools):
    def agent_node(state: AgentState) -> dict:
        llm = get_llm(temperature=0).bind_tools(tools)
        response: AIMessage = llm.invoke([SYSTEM, *state["messages"]])
        n_calls = len(response.tool_calls or [])
        preview = (response.content or "")[:80]
        print(f"  [agent] tool_calls={n_calls}  content={preview!r}")
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile()


async def run_once(question: str) -> str:
    tools = await load_mcp_tools()
    graph = build_graph(tools)
    show_graph(graph, title="MCP + LangGraph ReAct")
    print(f"=== Question ===\n  {question}\n")
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=question)]}
    )
    final = result["messages"][-1]
    text = final.content if isinstance(final.content, str) else str(final.content)
    print("=== Final answer ===")
    print(f"  {text}\n")
    return text


def main() -> None:
    require_ollama()
    if not SERVER.is_file():
        raise SystemExit(f"Missing MCP server file: {SERVER}")
    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is the demo quote for TCS.NS and the USDINR rate?"
    )
    asyncio.run(run_once(question))
    print("Next: optional live Yahoo MCP → 02_mcp_yahoo_optional.py\n")


if __name__ == "__main__":
    main()
