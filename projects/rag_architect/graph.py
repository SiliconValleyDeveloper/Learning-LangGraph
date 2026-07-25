"""LangGraph: strategy router → retrieve → (grade/retry) → generate → verify."""

from __future__ import annotations

import re
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from Learning.llm import get_llm
from projects.rag_architect.config import STRATEGIES, load_config
from projects.rag_architect.graph_rag import graph_search
from projects.rag_architect.models import ChunkHit, Strategy
from projects.rag_architect.retrieve import (
    dense_search,
    format_context,
    hybrid_search,
    hyde_search,
    lexical_rerank,
    multi_query,
)

REFUSE = (
    "I do not have enough grounded evidence in the Contoso Ops knowledge base "
    "to answer confidently."
)


class ArchitectState(TypedDict):
    question: str
    strategy: Strategy
    query: str
    hyde_passage: str
    hits: list[ChunkHit]
    grade: str
    retries: int
    answer: str
    verified: bool
    sources: list[str]
    notes: list[str]
    agent_plan: str


def _clean(text: object) -> str:
    raw = text if isinstance(text, str) else str(text)
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def _notes(state: ArchitectState, line: str) -> list[str]:
    notes = list(state.get("notes") or [])
    notes.append(line)
    return notes


def choose_strategy(state: ArchitectState) -> dict[str, Any]:
    strategy = state.get("strategy") or load_config().default_strategy
    if strategy not in STRATEGIES:
        strategy = "hybrid"
    return {
        "strategy": strategy,  # type: ignore[dict-item]
        "query": state.get("query") or state["question"],
        "notes": _notes(state, f"strategy={strategy}"),
    }


def agent_plan_node(state: ArchitectState) -> dict[str, Any]:
    """Agentic mode: pick a tool plan (kb_search | graph | escalate)."""
    question = state["question"]
    prompt = [
        SystemMessage(
            content=(
                "You are a retrieval planner for an enterprise KB.\n"
                "Choose ONE tool:\n"
                "- kb_search — normal hybrid document search\n"
                "- graph — multi-hop entities / tickets / roles\n"
                "- escalate — question is out of scope or needs a human\n"
                "Reply with ONLY one word: kb_search | graph | escalate"
            )
        ),
        HumanMessage(content=question),
    ]
    plan = _clean(get_llm(temperature=0).invoke(prompt).content).lower()
    if "escalat" in plan:
        plan = "escalate"
    elif "graph" in plan:
        plan = "graph"
    else:
        plan = "kb_search"
    return {"agent_plan": plan, "notes": _notes(state, f"agent_plan={plan}")}


def retrieve_node(state: ArchitectState) -> dict[str, Any]:
    strategy: Strategy = state["strategy"]
    question = state["question"]
    query = state.get("query") or question
    cfg = load_config()
    notes_extra: list[str] = []
    hyde_text = ""

    if strategy == "agentic" and state.get("agent_plan") == "escalate":
        return {
            "hits": [],
            "grade": "fail",
            "notes": _notes(state, "retrieve_skipped=escalate"),
        }

    if strategy == "baseline":
        hits = dense_search(query, k=cfg.top_k)
    elif strategy == "hyde":
        hits, hyde_text = hyde_search(question, k=cfg.top_k)
        notes_extra.append(f"hyde_chars={len(hyde_text)}")
    elif strategy == "graph" or (
        strategy == "agentic" and state.get("agent_plan") == "graph"
    ):
        hits, gnotes = graph_search(question, k=cfg.top_k)
        notes_extra.extend(gnotes)
    else:
        # hybrid, crag, agentic kb_search
        if strategy == "crag" and state.get("retries", 0) > 0:
            alts = multi_query(question)
            query = alts[0] if alts else query
            notes_extra.append(f"crag_rewrite={query}")
        hits = hybrid_search(query, k=cfg.retrieve_candidates)
        hits = lexical_rerank(question, hits, k=cfg.top_k)

    sources = list(dict.fromkeys(h.source for h in hits))
    note_lines = list(state.get("notes") or [])
    note_lines.append(f"retrieve_hits={len(hits)}")
    note_lines.extend(notes_extra)
    return {
        "hits": hits,
        "query": query,
        "hyde_passage": hyde_text,
        "sources": sources,
        "notes": note_lines,
    }


def grade_node(state: ArchitectState) -> dict[str, Any]:
    if not state.get("hits"):
        return {"grade": "fail", "notes": _notes(state, "grade=fail(empty)")}

    # LLM grader for CRAG / agentic; cheap non-empty check for other strategies.
    if state["strategy"] not in {"crag", "agentic"}:
        return {"grade": "pass", "notes": _notes(state, "grade=pass(skip_llm)")}

    context = format_context(state["hits"])
    prompt = [
        SystemMessage(
            content=(
                "Grade whether CONTEXT can answer QUESTION. "
                "Reply ONLY: pass OR fail."
            )
        ),
        HumanMessage(content=f"QUESTION:\n{state['question']}\n\nCONTEXT:\n{context}"),
    ]
    label = _clean(get_llm(temperature=0).invoke(prompt).content).lower()
    grade = "pass" if label.startswith("pass") else "fail"
    if "pass" in label and "fail" not in label:
        grade = "pass"
    elif "fail" in label:
        grade = "fail"
    return {"grade": grade, "notes": _notes(state, f"grade={grade}")}


def rewrite_node(state: ArchitectState) -> dict[str, Any]:
    alts = multi_query(state["question"])
    query = alts[0] if alts else state["question"]
    return {
        "query": query,
        "retries": state.get("retries", 0) + 1,
        "notes": _notes(state, f"rewrite → {query}"),
    }


def generate_node(state: ArchitectState) -> dict[str, Any]:
    if state.get("agent_plan") == "escalate":
        return {
            "answer": (
                "This looks out of scope for the Contoso Ops knowledge base. "
                "Please escalate to #ops-oncall or your manager."
            ),
            "verified": True,
        }

    if not state.get("hits") or state.get("grade") == "fail":
        return {"answer": REFUSE, "verified": True}

    context = format_context(state["hits"])
    prompt = [
        SystemMessage(
            content=(
                "You are Contoso Ops KB assistant. Answer using ONLY the context. "
                "Cite sources as [1], [2]. If insufficient, say you do not know."
            )
        ),
        HumanMessage(content=f"QUESTION:\n{state['question']}\n\nCONTEXT:\n{context}"),
    ]
    answer = _clean(get_llm(temperature=0.2).invoke(prompt).content)
    return {"answer": answer}


def verify_node(state: ArchitectState) -> dict[str, Any]:
    answer = state.get("answer") or ""
    if answer == REFUSE or "escalate" in (state.get("agent_plan") or ""):
        return {"verified": True, "notes": _notes(state, "verify=ok(refuse_or_escalate)")}

    has_cite = bool(re.search(r"\[\d+\]", answer))
    context = format_context(state.get("hits") or []).lower()
    # crude faithfulness: majority of content words from answer appear in context
    tokens = [
        t
        for t in re.findall(r"[a-z0-9]{4,}", answer.lower())
        if t not in {"that", "this", "with", "from", "have", "been", "contoso"}
    ]
    if not tokens:
        verified = has_cite
    else:
        overlap = sum(1 for t in tokens if t in context) / len(tokens)
        verified = has_cite and overlap >= 0.35

    if not verified and state.get("hits"):
        # one repair attempt inline
        prompt = [
            SystemMessage(
                content=(
                    "Rewrite the answer so every claim is grounded in CONTEXT and "
                    "includes citations like [1]. If not possible, say you do not know."
                )
            ),
            HumanMessage(
                content=(
                    f"QUESTION:\n{state['question']}\n\nCONTEXT:\n"
                    f"{format_context(state['hits'])}\n\nDRAFT:\n{answer}"
                )
            ),
        ]
        answer = _clean(get_llm(temperature=0).invoke(prompt).content)
        has_cite = bool(re.search(r"\[\d+\]", answer))
        verified = has_cite or answer.startswith("I do not")
        return {
            "answer": answer,
            "verified": verified,
            "notes": _notes(state, f"verify_repair cited={has_cite}"),
        }

    return {"verified": verified, "notes": _notes(state, f"verified={verified}")}


def route_after_choose(
    state: ArchitectState,
) -> Literal["agent_plan", "retrieve"]:
    if state["strategy"] == "agentic":
        return "agent_plan"
    return "retrieve"


def route_after_grade(
    state: ArchitectState,
) -> Literal["generate", "rewrite"]:
    cfg = load_config()
    if state.get("grade") == "pass":
        return "generate"
    if state["strategy"] != "crag":
        return "generate"
    if state.get("retries", 0) >= cfg.max_crag_retries:
        return "generate"
    return "rewrite"


def build_graph():
    g = StateGraph(ArchitectState)
    g.add_node("choose_strategy", choose_strategy)
    g.add_node("agent_plan", agent_plan_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("generate", generate_node)
    g.add_node("verify", verify_node)

    g.add_edge(START, "choose_strategy")
    g.add_conditional_edges(
        "choose_strategy",
        route_after_choose,
        {"agent_plan": "agent_plan", "retrieve": "retrieve"},
    )
    g.add_edge("agent_plan", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "verify")
    g.add_edge("verify", END)
    return g.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def reset_graph() -> None:
    global _GRAPH
    _GRAPH = None
