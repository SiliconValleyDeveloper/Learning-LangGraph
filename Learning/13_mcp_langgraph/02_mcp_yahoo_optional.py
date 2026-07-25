"""
Phase 13 · Lesson 2 — Optional: wire Cursor's Yahoo Finance MCP into LangGraph

Uses the same MCP package Cursor uses:
  npx -y @modelcontextprotocol/server-yahoo-finance@latest

Needs:
  - Node.js / npx on PATH
  - Network (live Yahoo data)
  - Ollama running
  - pip: langchain-mcp-adapters

Run:
    python Learning/13_mcp_langgraph/02_mcp_yahoo_optional.py
    python Learning/13_mcp_langgraph/02_mcp_yahoo_optional.py "What is AAPL trading at?"
"""

from __future__ import annotations

import asyncio
import shutil
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


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM = SystemMessage(
    content=(
        "You are a markets assistant. Use Yahoo Finance MCP tools for live data. "
        "Cite the tool results. If a tool fails, say so — do not invent prices. "
        "Keep answers short."
    )
)


def _yahoo_connection() -> dict:
    npx = shutil.which("npx")
    if not npx:
        raise SystemExit(
            "npx not found. Install Node.js, or stick to Lesson 1 "
            "(01_mcp_tools_agent.py) which needs no network."
        )
    return {
        "yahoo_finance": {
            "command": npx,
            "args": ["-y", "@modelcontextprotocol/server-yahoo-finance@latest"],
            "transport": "stdio",
        }
    }


async def load_yahoo_tools():
    from langchain_mcp_adapters.client import MultiServerMCPClient

    print("  [mcp] starting Yahoo Finance MCP via npx (may download on first run)…")
    client = MultiServerMCPClient(_yahoo_connection())
    tools = await client.get_tools()
    print(f"  [mcp] loaded {len(tools)} tool(s): {', '.join(t.name for t in tools)}")
    return tools


def build_graph(tools):
    def agent_node(state: AgentState) -> dict:
        llm = get_llm(temperature=0).bind_tools(tools)
        response: AIMessage = llm.invoke([SYSTEM, *state["messages"]])
        n_calls = len(response.tool_calls or [])
        print(f"  [agent] tool_calls={n_calls}")
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile()


async def run_once(question: str) -> str:
    tools = await load_yahoo_tools()
    if not tools:
        raise SystemExit("Yahoo MCP returned no tools — check network / npx.")
    graph = build_graph(tools)
    show_graph(graph, title="Yahoo MCP + LangGraph")
    result = await graph.ainvoke({"messages": [HumanMessage(content=question)]})
    final = result["messages"][-1]
    text = final.content if isinstance(final.content, str) else str(final.content)
    print("\n=== Final answer ===")
    print(text)
    return text


def main() -> None:
    require_ollama()
    question = sys.argv[1] if len(sys.argv) > 1 else "What is the latest price of AAPL?"
    asyncio.run(run_once(question))


if __name__ == "__main__":
    main()
