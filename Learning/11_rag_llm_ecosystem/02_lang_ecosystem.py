"""
Phase 11 · Lesson 2 — LangChain, LangGraph, and LangSmith

What you will learn
-------------------
1. LangChain composes prompts, models, parsers, documents, and retrievers
2. LCEL is concise for a fixed linear pipeline
3. LangGraph is explicit when state, branches, loops, or persistence matter
4. LangSmith traces and evaluates either style (not required for this lesson)

Run:
    python Learning/11_rag_llm_ecosystem/02_lang_ecosystem.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import get_llm, require_ollama
from visualize import show_graph


def build_chain():
    """LCEL: prompt → model → string parser."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Teach the Lang ecosystem in at most two sentences."),
            ("human", "{question}"),
        ]
    )
    return prompt | get_llm(temperature=0) | StrOutputParser()


class EcosystemState(TypedDict):
    question: str
    answer: str


def explain_node(state: EcosystemState) -> dict:
    """A graph node can internally reuse the same LangChain components."""
    answer = build_chain().invoke({"question": state["question"]})
    return {"answer": answer}


def build_graph():
    """LangGraph: explicit state + node + edges."""
    builder = StateGraph(EcosystemState)
    builder.add_node("explain", explain_node)
    builder.add_edge(START, "explain")
    builder.add_edge("explain", END)
    return builder.compile()


if __name__ == "__main__":
    require_ollama()
    question = "When should I use LangChain versus LangGraph?"

    print("=== LangChain LCEL ===")
    print(build_chain().invoke({"question": question}))

    print("\n=== Same work inside LangGraph ===")
    graph = build_graph()
    show_graph(graph, title="Lang ecosystem")
    result = graph.invoke({"question": question, "answer": ""})
    print(result["answer"])

    print(
        "\nRule of thumb: use LangChain building blocks inside LangGraph when "
        "the application needs visible state and control flow. LangSmith can "
        "trace both, but is not needed to run them."
    )
