"""
Phase 12 · Lesson 5 — Corrective RAG (CRAG) / Self-RAG control loop

Standard RAG: retrieve → generate
CRAG:        retrieve → grade → if bad, rewrite/correct → generate
Self-RAG:    model decides when to retrieve / critique its own draft

Graph:
    START → retrieve → grade → (pass) → generate → END
                         └─ (fail & retries) → rewrite → retrieve …

Needs: Ollama with qwen3:8b and nomic-embed-text

Run:
    python Learning/12_rag_architect/05_crag_self_rag.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from helpers import format_context, hybrid_search, short_preview
from llm import get_llm, require_ollama
from visualize import show_graph

MAX_RETRIES = 2


class CRAGState(TypedDict):
    question: str
    query: str
    documents: list[Document]
    grade: str
    retries: int
    answer: str
    notes: list[str]


def _clean(text: object) -> str:
    raw = text if isinstance(text, str) else str(text)
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def retrieve_node(state: CRAGState) -> dict:
    query = state.get("query") or state["question"]
    docs = hybrid_search(query, k=4)
    notes = list(state.get("notes") or [])
    notes.append(f"retrieve[{query}] → {len(docs)} chunks")
    return {"documents": docs, "query": query, "notes": notes}


def grade_node(state: CRAGState) -> dict:
    context = format_context(state["documents"])
    prompt = [
        SystemMessage(
            content=(
                "Grade whether the CONTEXT can answer the QUESTION. "
                "Reply ONLY: pass OR fail."
            )
        ),
        HumanMessage(content=f"QUESTION:\n{state['question']}\n\nCONTEXT:\n{context}"),
    ]
    label = _clean(get_llm(temperature=0).invoke(prompt).content).lower()
    grade = "pass" if "pass" in label and "fail" not in label.split()[0:1] else (
        "pass" if label.startswith("pass") else "fail"
    )
    if "pass" in label and "fail" not in label:
        grade = "pass"
    elif "fail" in label:
        grade = "fail"
    else:
        grade = "fail"
    notes = list(state.get("notes") or [])
    notes.append(f"grade={grade}")
    return {"grade": grade, "notes": notes}


def rewrite_node(state: CRAGState) -> dict:
    prompt = [
        SystemMessage(
            content=(
                "The previous search failed. Rewrite ONE better enterprise KB "
                "search query. Return only the query."
            )
        ),
        HumanMessage(content=state["question"]),
    ]
    query = _clean(get_llm(temperature=0.2).invoke(prompt).content).splitlines()[0]
    notes = list(state.get("notes") or [])
    notes.append(f"rewrite → {query}")
    return {"query": query, "retries": state.get("retries", 0) + 1, "notes": notes}


def generate_node(state: CRAGState) -> dict:
    context = format_context(state["documents"])
    if state.get("grade") == "fail":
        answer = (
            "I do not have enough grounded evidence in the Contoso Ops knowledge "
            "base to answer confidently."
        )
        return {"answer": answer}
    prompt = [
        SystemMessage(
            content=(
                "Answer using ONLY the context. Cite sources like [1], [2]. "
                "If context is insufficient, say you do not know."
            )
        ),
        HumanMessage(content=f"QUESTION:\n{state['question']}\n\nCONTEXT:\n{context}"),
    ]
    answer = _clean(get_llm(temperature=0.2).invoke(prompt).content)
    return {"answer": answer}


def route_after_grade(state: CRAGState) -> Literal["generate", "rewrite"]:
    if state.get("grade") == "pass":
        return "generate"
    if state.get("retries", 0) >= MAX_RETRIES:
        return "generate"
    return "rewrite"


def build_graph():
    g = StateGraph(CRAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("generate", generate_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade, {
        "generate": "generate",
        "rewrite": "rewrite",
    })
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    return g.compile()


def main() -> None:
    require_ollama()
    graph = build_graph()
    show_graph(graph, title="CRAG / Self-RAG style loop")

    questions = [
        "What is the P1 acknowledge time?",
        "What is the capital of France?",  # should fail / refuse
    ]
    for question in questions:
        print("\n" + "=" * 60)
        print(f"Q: {question}")
        result = graph.invoke(
            {
                "question": question,
                "query": question,
                "documents": [],
                "grade": "",
                "retries": 0,
                "answer": "",
                "notes": [],
            }
        )
        for note in result["notes"]:
            print(f"  · {note}")
        print(f"A: {short_preview(result['answer'], 280)}")


if __name__ == "__main__":
    main()
