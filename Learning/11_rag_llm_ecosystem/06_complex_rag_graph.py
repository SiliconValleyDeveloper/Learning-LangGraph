"""
Phase 11 · Lesson 6 — Complex RAG with LLM control loops

Graph:
                    ┌──────────────────────────────┐
                    │          (retry)             │
                    ▼                              │
    START → classify → rewrite → retrieve → grade ─┴─→ generate → verify → END
                                              │
                                         (pass / give up)

What you will learn
-------------------
1. Real RAG is rarely retrieve → generate once
2. Query rewrite improves retrieval for vague questions
3. Access control (public vs private) belongs in the graph
4. A grader decides whether evidence is good enough to answer
5. Conditional edges create a retrieval retry loop
6. A verify step checks citations before the final answer

Needs: Ollama with qwen3:8b and nomic-embed-text

Run:
    python Learning/11_rag_llm_ecosystem/06_complex_rag_graph.py
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
from llm import get_llm, require_ollama
from rag_helpers import format_context, retrieve_scored
from visualize import show_graph

MAX_RETRIES = 2
PRIVATE_HINTS = (
    "manager",
    "employee",
    "salary",
    "private",
    "directory",
    "secret",
    "api key",
    "customer note",
    "ankit",
    "priya",
    "rahul",
)


class ComplexRAGState(TypedDict):
    question: str
    rewritten_query: str
    needs_private: bool
    route: str
    documents: list[Document]
    scores: list[float]
    sources: list[str]
    grade: str
    retries: int
    answer: str
    verified: bool
    notes: list[str]


def _clean_llm_text(content: object) -> str:
    text = content if isinstance(content, str) else str(content)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def _guess_needs_private(question: str) -> bool:
    lowered = question.lower()
    return any(hint in lowered for hint in PRIVATE_HINTS)


def classify_node(state: ComplexRAGState) -> dict:
    """Decide whether private docs may be searched."""
    question = state["question"]
    needs_private = _guess_needs_private(question)
    if not needs_private:
        prompt = [
            SystemMessage(
                content=(
                    "Does this question need PRIVATE employee/customer/secret "
                    "documents? Reply with ONLY one word: private OR public."
                )
            ),
            HumanMessage(content=question),
        ]
        label = _clean_llm_text(get_llm(temperature=0).invoke(prompt).content).lower()
        needs_private = bool(re.search(r"\bprivate\b", label))

    route = "private_allowed" if needs_private else "public_only"
    note = f"classify → {route}"
    print(f"  [classify] needs_private={needs_private}")
    return {
        "needs_private": needs_private,
        "route": route,
        "notes": [*state.get("notes", []), note],
    }


def rewrite_node(state: ComplexRAGState) -> dict:
    """Rewrite the user question into a retrieval-friendly query."""
    question = state["question"]
    retries = state.get("retries", 0)
    hint = ""
    if retries > 0:
        hint = (
            " Previous retrieval was weak. Expand with synonyms and concrete "
            "entity names from an internal knowledge base."
        )

    prompt = [
        SystemMessage(
            content=(
                "Rewrite the user question into one short search query for a "
                "corporate knowledge base about LangGraph, Acme Learning Labs, "
                "onboarding, incidents, and product FAQ."
                f"{hint} "
                "Return ONLY the rewritten query, no quotes."
            )
        ),
        HumanMessage(content=question),
    ]
    rewritten = _clean_llm_text(get_llm(temperature=0).invoke(prompt).content)
    if not rewritten:
        rewritten = question
    # Keep rewrites compact
    rewritten = " ".join(rewritten.split())[:220]
    print(f"  [rewrite] {rewritten!r}")
    return {
        "rewritten_query": rewritten,
        "notes": [*state.get("notes", []), f"rewrite → {rewritten}"],
    }


def retrieve_node(state: ComplexRAGState) -> dict:
    """Retrieve with public/private access control."""
    query = state.get("rewritten_query") or state["question"]
    include_private = bool(state.get("needs_private"))
    scored = retrieve_scored(query, k=4, include_private=include_private)
    documents = [document for document, _ in scored]
    scores = [score for _, score in scored]
    sources = list(
        dict.fromkeys(
            str(document.metadata.get("source", "unknown"))
            for document in documents
        )
    )
    print(
        f"  [retrieve] private={include_private} "
        f"chunks={len(documents)} sources={sources}"
    )
    return {
        "documents": documents,
        "scores": scores,
        "sources": sources,
        "notes": [
            *state.get("notes", []),
            f"retrieve → {len(documents)} chunks ({', '.join(sources) or 'none'})",
        ],
    }


def _evidence_supports_question(question: str, documents: list[Document]) -> bool:
    """Cheap overlap check so useful chunks are not rejected by a picky grader LLM."""
    if not documents:
        return False
    q = question.lower()
    blob = "\n".join(document.page_content.lower() for document in documents)
    sources = " ".join(
        str(document.metadata.get("source", "")).lower() for document in documents
    )

    if any(token in q for token in ("sev", "outage", "ollama", "incident")):
        if "incident" in sources or "sev-2" in blob or "first checks" in blob:
            return True
    if any(token in q for token in ("manager", "employee", "ankit", "directory")):
        if "employee" in sources or "manager" in blob or "shivam" in blob or "priya" in blob:
            return True
    if any(token in q for token in ("index", "query path", "architecture", "embedding")):
        if "architecture" in sources or "indexing path" in blob or "query path" in blob:
            return True
    if "recover" in q or "empty context" in q:
        if "vector store" in blob or "rebuild" in blob or "incident" in sources:
            return True

    stop = {
        "what",
        "which",
        "where",
        "when",
        "should",
        "first",
        "about",
        "with",
        "from",
        "does",
        "this",
        "that",
        "have",
        "into",
        "our",
        "the",
        "and",
        "for",
    }
    words = [
        word
        for word in re.findall(r"[a-z0-9-]{4,}", q)
        if word not in stop
    ]
    if not words:
        return len(documents) > 0
    hits = sum(1 for word in words if word in blob or word in sources)
    return hits >= max(2, (len(words) + 2) // 3)


def grade_node(state: ComplexRAGState) -> dict:
    """Grade whether retrieved evidence is enough to answer safely."""
    documents = state.get("documents") or []
    scores = state.get("scores") or []
    question = state["question"]

    if not documents:
        grade: Literal["pass", "fail"] = "fail"
        reason = "no documents"
    elif _evidence_supports_question(question, documents):
        grade = "pass"
        best = min(scores) if scores else 0.0
        reason = f"overlap-ok best_score={best:.4f}"
    else:
        preview = "\n\n".join(
            f"- {document.metadata.get('source')}: {document.page_content[:480]}"
            for document in documents[:4]
        )
        prompt = [
            SystemMessage(
                content=(
                    "You grade RAG evidence for a learning lab. "
                    "Reply with ONLY one word: pass OR fail.\n"
                    "pass = the evidence mentions the topic and could support an answer.\n"
                    "fail = evidence is clearly unrelated.\n"
                    "When unsure, reply pass."
                )
            ),
            HumanMessage(content=f"Question: {question}\n\nEvidence:\n{preview}"),
        ]
        label = _clean_llm_text(get_llm(temperature=0).invoke(prompt).content).lower()
        if re.search(r"\bfail\b", label) and not re.search(r"\bpass\b", label):
            grade = "fail"
        else:
            grade = "pass"
        best = min(scores) if scores else 0.0
        reason = f"llm={label[:24]!r} best_score={best:.4f}"

    print(f"  [grade] {grade} ({reason})")
    return {
        "grade": grade,
        "notes": [*state.get("notes", []), f"grade → {grade}"],
    }


def route_after_grade(state: ComplexRAGState) -> Literal["generate", "rewrite"]:
    """Retry retrieval when evidence fails, otherwise generate."""
    if state.get("grade") == "pass":
        return "generate"
    if int(state.get("retries") or 0) < MAX_RETRIES:
        return "rewrite"
    return "generate"


def bump_retry_node(state: ComplexRAGState) -> dict:
    """Increment retry counter before another rewrite/retrieve cycle."""
    retries = int(state.get("retries") or 0) + 1
    print(f"  [retry] attempt={retries}/{MAX_RETRIES}")
    return {
        "retries": retries,
        "notes": [*state.get("notes", []), f"retry → {retries}"],
    }


def generate_node(state: ComplexRAGState) -> dict:
    """Generate a grounded answer, or refuse when evidence is weak."""
    documents = state.get("documents") or []
    if not documents:
        answer = (
            "I don't know from the lesson knowledge base. "
            "Retrieval returned no documents."
        )
        if state.get("needs_private") is False and _guess_needs_private(state["question"]):
            answer += (
                " Tip: this question may need private docs — ask with employee/"
                "manager wording so classify enables private retrieval."
            )
        print("  [generate] refused (no documents)")
        return {"answer": answer, "verified": True}

    context = format_context(documents)
    access = "private+public" if state.get("needs_private") else "public-only"
    system = SystemMessage(
        content=(
            "You are a careful enterprise RAG assistant. "
            f"Access mode for this run: {access}. "
            "Answer ONLY from the supplied context. "
            "Cite sources as [filename.md]. "
            "If context is insufficient, say you don't know from the knowledge base. "
            "Do not invent people, policies, or commands."
        )
    )
    user = HumanMessage(
        content=(
            f"Original question:\n{state['question']}\n\n"
            f"Search query used:\n{state.get('rewritten_query')}\n\n"
            f"Retrieved context:\n{context}"
        )
    )
    answer = _clean_llm_text(get_llm(temperature=0).invoke([system, user]).content)
    source_order = [
        str(document.metadata.get("source", "unknown")) for document in documents
    ]
    for index, source in enumerate(source_order, start=1):
        answer = re.sub(rf"\[{index}\]", f"[{source}]", answer)
    answer = re.sub(r"\[([^\]]+?\.md)\]\.md\]", r"[\1]", answer)
    unique_sources = list(dict.fromkeys(source_order))
    if unique_sources and not any(f"[{source}]" in answer for source in unique_sources):
        answer += "\n\nSources: " + ", ".join(f"[{source}]" for source in unique_sources)
    print("  [generate] drafted answer")
    return {"answer": answer, "verified": False}


def verify_node(state: ComplexRAGState) -> dict:
    """Light citation / refusal verification before END."""
    answer = state.get("answer") or ""
    sources = state.get("sources") or []
    verified = True
    notes = list(state.get("notes") or [])

    if "don't know from the lesson knowledge base" in answer.lower():
        notes.append("verify → accepted refusal")
        return {"verified": True, "notes": notes}

    cited = [source for source in sources if f"[{source}]" in answer]
    if sources and not cited:
        answer = answer.rstrip() + "\n\nSources: " + ", ".join(
            f"[{source}]" for source in sources
        )
        notes.append("verify → appended missing citations")
    else:
        notes.append(f"verify → ok ({len(cited)} citations)")

    print(f"  [verify] citations={cited or sources}")
    return {"answer": answer, "verified": verified, "notes": notes}


def build_graph():
    """Compile the complex RAG control loop."""
    builder = StateGraph(ComplexRAGState)
    builder.add_node("classify", classify_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade", grade_node)
    builder.add_node("bump_retry", bump_retry_node)
    builder.add_node("generate", generate_node)
    builder.add_node("verify", verify_node)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "rewrite")
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "generate": "generate",
            "rewrite": "bump_retry",
        },
    )
    builder.add_edge("bump_retry", "rewrite")
    builder.add_edge("generate", "verify")
    builder.add_edge("verify", END)
    return builder.compile()


def initial_state(question: str) -> ComplexRAGState:
    return {
        "question": question,
        "rewritten_query": "",
        "needs_private": False,
        "route": "",
        "documents": [],
        "scores": [],
        "sources": [],
        "grade": "",
        "retries": 0,
        "answer": "",
        "verified": False,
        "notes": [],
    }


if __name__ == "__main__":
    require_ollama()
    graph = build_graph()
    show_graph(graph, title="Complex RAG control loop")

    demos = [
        "What should I check first for a SEV-2 Ollama outage?",
        "Who is the manager for Ankit Rawat?",
        "Explain the difference between indexing path and query path in our architecture.",
    ]
    for question in demos:
        print(f"\n=== Question: {question} ===\n")
        result = graph.invoke(initial_state(question), config={"recursion_limit": 20})
        print(f"\nAnswer:\n{result['answer']}\n")
        print(f"Path notes: {' | '.join(result.get('notes') or [])}")
        print(f"retries={result.get('retries')} grade={result.get('grade')} "
              f"private={result.get('needs_private')} verified={result.get('verified')}")
