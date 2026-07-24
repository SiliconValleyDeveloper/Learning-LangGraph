"""
Phase 11 · Lesson 5 — A complete RAG graph

Graph:
    START → retrieve → generate → END

What you will learn
-------------------
1. Retrieval and generation are separate, observable nodes
2. Retrieved documents travel through graph state
3. The prompt limits the LLM to supplied context
4. Source metadata makes the answer auditable

Run:
    python Learning/11_rag_llm_ecosystem/05_rag_graph.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import get_llm, require_ollama
from rag_helpers import format_context, retrieve
from visualize import show_graph


class RAGState(TypedDict):
    question: str
    documents: list[Document]
    sources: list[str]
    answer: str


def retrieve_node(state: RAGState) -> dict:
    """Find chunks semantically related to the question."""
    documents = retrieve(state["question"], k=3)
    sources = list(
        dict.fromkeys(
            str(document.metadata.get("source", "unknown"))
            for document in documents
        )
    )
    return {"documents": documents, "sources": sources}


def generate_node(state: RAGState) -> dict:
    """Generate an answer using only retrieved context."""
    documents = state.get("documents") or []
    if not documents:
        return {
            "answer": (
                "I don't know from the lesson knowledge base because retrieval "
                "returned no relevant documents."
            )
        }

    context = format_context(documents)
    system = SystemMessage(
        content=(
            "You are a grounded RAG tutor. Answer ONLY from the supplied context. "
            "Do not use outside knowledge. If the context is insufficient, say "
            "\"I don't know from the lesson knowledge base.\" "
            "Cite factual claims with the source filename in square brackets, "
            "for example [rag_guide.md]. Keep the answer concise."
        )
    )
    user = HumanMessage(
        content=f"Question:\n{state['question']}\n\nRetrieved context:\n{context}"
    )
    response = get_llm(temperature=0).invoke([system, user])
    answer = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )
    source_order = [
        str(document.metadata.get("source", "unknown"))
        for document in documents
    ]
    for index, source in enumerate(source_order, start=1):
        answer = re.sub(rf"\[{index}\]", f"[{source}]", answer)

    unique_sources = list(dict.fromkeys(source_order))
    if unique_sources and not any(f"[{source}]" in answer for source in unique_sources):
        answer += "\n\nSources: " + ", ".join(
            f"[{source}]" for source in unique_sources
        )
    return {"answer": answer}


def build_graph():
    """Compile the two-node RAG workflow used by CLI and visual lab."""
    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()


def initial_state(question: str) -> RAGState:
    return {
        "question": question,
        "documents": [],
        "sources": [],
        "answer": "",
    }


if __name__ == "__main__":
    require_ollama()
    graph = build_graph()
    show_graph(graph, title="RAG: retrieve then generate")

    question = "What is a LangGraph node, and how is it connected?"
    print(f"\nQuestion: {question}\n")

    for update in graph.stream(initial_state(question), stream_mode="updates"):
        node, payload = next(iter(update.items()))
        if node == "retrieve":
            print(f"[retrieve] sources: {payload['sources']}")
        elif node == "generate":
            print(f"[generate]\n{payload['answer']}")
