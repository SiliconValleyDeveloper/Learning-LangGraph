"""
Phase 9 · Lesson 1 — ReAct agent with live internet search

What you will learn
-------------------
1. Wrap an internet search provider as a LangChain tool
2. Let an agent decide when a question needs fresh web information
3. Feed titles, snippets, and URLs back through a ToolMessage
4. Require the final answer to cite the sources it actually received

Needs: Ollama running and an internet connection

Run:
    python Learning/09_web_search/01_search_agent.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, TypedDict

from ddgs import DDGS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import get_llm, require_ollama
from visualize import show_graph


@tool
def internet_search(query: str) -> str:
    """Search the live internet and return up to five titled results with URLs."""
    cleaned = query.strip()
    if not cleaned:
        return "Error: search query cannot be empty."

    try:
        results = list(DDGS().text(cleaned, max_results=5))
    except Exception as exc:  # noqa: BLE001 — return a recoverable tool error
        return f"Error: internet search failed: {exc}"

    if not results:
        return f"No web results found for: {cleaned}"

    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        title = result.get("title") or "Untitled result"
        url = result.get("href") or result.get("url") or ""
        snippet = result.get("body") or result.get("snippet") or ""
        lines.append(f"{index}. {title}\nURL: {url}\nSnippet: {snippet}")
    return "\n\n".join(lines)


TOOLS = [internet_search]


class SearchState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM = SystemMessage(
    content=(
        "You are a web research assistant. Use internet_search for current facts, "
        "news, releases, prices, or whenever the user asks you to search the web. "
        "Answer only from the returned results. End with a Sources section containing "
        "the exact source URLs you used. If search fails, explain the failure instead "
        "of inventing an answer."
    )
)


def search_agent(state: SearchState) -> dict:
    llm = get_llm(temperature=0).bind_tools(TOOLS)
    response: AIMessage = llm.invoke([SYSTEM, *state["messages"]])
    return {"messages": [response]}


def build_graph(checkpointer=None):
    builder = StateGraph(SearchState)
    builder.add_node("search_agent", search_agent)
    builder.add_node("internet_search", ToolNode(TOOLS))
    builder.add_edge(START, "search_agent")
    builder.add_conditional_edges(
        "search_agent",
        tools_condition,
        {"tools": "internet_search", "__end__": "__end__"},
    )
    builder.add_edge("internet_search", "search_agent")
    return builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    require_ollama()
    graph = build_graph()
    show_graph(graph, title="Internet search agent")
    result = graph.invoke(
        {"messages": [HumanMessage(content="Search the web for the latest LangGraph release.")]}
    )
    print(f"\n{result['messages'][-1].content}\n")
