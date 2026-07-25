"""Advanced chatbot: understand intent → chat | docs | web | hybrid → grounded answer."""

from __future__ import annotations

import re
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from Learning.llm import get_llm
from projects.advanced_chatbot.rerank import (
    build_context_from_ranked,
    rerank_chunks,
    retrieve_candidate_count,
)
from projects.advanced_chatbot.store import get_store
from projects.advanced_chatbot.web_search import (
    build_search_queries,
    evidence_overlap,
    format_web_context,
    search_web,
)

Intent = Literal["chat", "documents", "web", "hybrid"]

# Clear small-talk — never pull from uploaded docs / web.
_CHAT_EXACT = {
    "hi",
    "hello",
    "hey",
    "howdy",
    "yo",
    "sup",
    "hi there",
    "hello there",
    "hey there",
    "hi how are you",
    "hello how are you",
    "hey how are you",
    "hi how are you doing",
    "hello how are you doing",
    "how are you",
    "how are you doing",
    "hows it going",
    "how's it going",
    "whats up",
    "what's up",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "thanks",
    "thank you",
    "ty",
    "bye",
    "goodbye",
    "see ya",
    "ok",
    "okay",
    "cool",
    "nice",
    "great",
    "awesome",
    "who are you",
    "what can you do",
    "help",
    "help me",
}


# Capability / meta asks — answer with LLM ("yes I can…"), do NOT run tools yet.
_CAPABILITY_RE = re.compile(
    r"^\s*("
    r"(can|could|will|would|do|are)\s+you\s+"
    r"(search|look\s+up|browse|use|access|check|find|go\s+online|"
    r"search\s+(on\s+)?(the\s+)?(internet|web)|"
    r"use\s+(the\s+)?(internet|web)|"
    r"read\s+(my\s+)?(documents?|files?|pdfs?)|"
    r"answer\s+(from|using)\s+(documents?|files?)|"
    r"help(\s+me)?)"
    r"|"
    r"(is\s+it\s+possible\s+(for\s+you\s+)?to\s+search|"
    r"are\s+you\s+able\s+to\s+(search|look\s+up|browse)|"
    r"do\s+you\s+(support|have)\s+(web\s+)?search|"
    r"can\s+you\s+search\s+(on\s+)?(the\s+)?(internet|web)\s*\??)"
    r").*$",
    re.IGNORECASE | re.DOTALL,
)

# Actual search command / topic (not just asking if we can).
_WEB_ACTION_RE = re.compile(
    r"\b("
    r"search\s+(the\s+)?(web|internet)\s+(for|about)|"
    r"look\s+up|google\s+|find\s+online|"
    r"what\s+is\s+the\s+latest|latest\s+|current\s+|today'?s\s+|"
    r"news\s+about|weather\s+in|stock\s+price\s+of|"
    r"who\s+won|election\s+results"
    r")\b",
    re.IGNORECASE,
)


def looks_like_chat(question: str) -> bool:
    """Fast heuristic for greetings / small talk / capability questions (no tools)."""
    raw = (question or "").strip()
    if not raw or len(raw) > 160:
        return False
    cleaned = re.sub(r"[^\w\s']+", " ", raw.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned in _CHAT_EXACT:
        return True
    return looks_like_capability_question(raw)


def looks_like_capability_question(question: str) -> bool:
    """True when user asks *if* we can search/read — not *to* search a topic."""
    text = (question or "").strip()
    if not text or len(text) > 160:
        return False
    # If they already give a search topic after "search … for X", it's a real web ask.
    if _WEB_ACTION_RE.search(text):
        return False
    if _CAPABILITY_RE.match(text):
        return True
    cleaned = re.sub(r"[^\w\s']+", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    capability_exact = {
        "could you search on the internet",
        "can you search on the internet",
        "can you search the internet",
        "could you search the internet",
        "can you search the web",
        "could you search the web",
        "can you search online",
        "do you search the web",
        "can you use the internet",
        "can you browse the internet",
        "are you able to search the web",
        "can you read my documents",
        "can you read uploaded files",
        "what can you search",
    }
    return cleaned in capability_exact


_DOC_HINT_RE = re.compile(
    r"\b("
    r"document|uploaded|pdf|file|screenshot|ocr|"
    r"according\s+to|in\s+the\s+(doc|document|pdf|file)|"
    r"summarize|summary|this\s+(doc|document|pdf|file)|"
    r"from\s+the\s+(doc|document|pdf|file)"
    r")\b",
    re.IGNORECASE,
)

_WEB_HINT_RE = re.compile(
    r"\b("
    r"search\s+(the\s+)?(web|internet)\s+(for|about)|"
    r"look\s+up|google\s+|find\s+online|"
    r"latest|current|today|yesterday|this\s+week|"
    r"news|breaking|live|weather|stock\s+price|"
    r"who\s+won|score|election"
    r")\b",
    re.IGNORECASE,
)


class ChatState(TypedDict):
    workspace_id: str
    question: str
    use_web_search: bool
    intent: str
    route_reason: str
    rewritten_query: str
    search_queries: list[str]
    context: str
    web_context: str
    sources: list[str]
    web_results: list[dict[str, str]]
    chunk_candidates: list[dict[str, Any]]
    chunk_previews: list[dict[str, Any]]
    rerank_backend: str
    doc_score: float
    web_score: float
    evidence_grade: str
    answer: str
    verified: bool
    fix_attempts: int


def _normalize_intent(raw: str) -> Intent:
    value = (raw or "").strip().lower()
    if value in {"chat", "documents", "web", "hybrid"}:
        return value  # type: ignore[return-value]
    if "chat" in value or "small" in value or "greet" in value:
        return "chat"
    if "hybrid" in value or "both" in value:
        return "hybrid"
    if "web" in value or "internet" in value or "search" in value:
        return "web"
    if "doc" in value or "rag" in value or "file" in value:
        return "documents"
    return "documents"


def _understand(state: ChatState) -> dict[str, Any]:
    """Decide which tool path to use: chat LLM, documents, web, or both."""
    question = (state.get("question") or "").strip()
    force_web = bool(state.get("use_web_search"))

    if looks_like_capability_question(question):
        return {
            "intent": "chat",
            "route_reason": (
                "Capability question (e.g. 'can you search?') — "
                "LLM explains yes/how; do not run search yet."
            ),
        }

    if looks_like_chat(question):
        return {
            "intent": "chat",
            "route_reason": "Greeting / small talk — answer with LLM only (ignore uploaded docs).",
        }

    # Cheap hints before calling the LLM.
    has_doc_hint = bool(_DOC_HINT_RE.search(question))
    has_web_hint = bool(_WEB_HINT_RE.search(question) or _WEB_ACTION_RE.search(question))
    if has_doc_hint and (has_web_hint or force_web):
        return {
            "intent": "hybrid",
            "route_reason": "Question needs documents and live/web context.",
        }
    if has_doc_hint and not force_web:
        return {
            "intent": "documents",
            "route_reason": "Question refers to uploaded document content.",
        }
    if has_web_hint or force_web:
        intent: Intent = "hybrid" if has_doc_hint else "web"
        return {
            "intent": intent,
            "route_reason": (
                "Search toggle on or web-oriented question — use internet tool."
                if force_web
                else "Question needs live/current information from the web."
            ),
        }

    llm = get_llm(temperature=0, reasoning=False)
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Classify the user message into exactly one intent label.\n"
                    "Labels:\n"
                    "- chat — greetings, thanks, small talk, OR capability questions "
                    "like 'could you search on the internet?', 'can you read my docs?' "
                    "(asking IF you can, not asking you TO search a topic yet)\n"
                    "- documents — asks about uploaded files/PDFs/screenshots or private doc facts\n"
                    "- web — asks you TO look up a live/current public fact "
                    "(e.g. 'search the web for AWS exam tips')\n"
                    "- hybrid — needs both uploaded docs AND live web\n\n"
                    "Rules:\n"
                    "- 'Could you search on the internet?' with NO topic → chat\n"
                    "- 'Search the internet for X' / 'latest news about X' → web\n"
                    "- If they only greet, choose chat (even if documents exist).\n"
                    "- Return ONLY one word: chat | documents | web | hybrid"
                )
            ),
            HumanMessage(content=question),
        ]
    )
    intent = _normalize_intent(str(response.content))
    if force_web and intent == "documents":
        intent = "hybrid"
    if force_web and intent == "chat" and not looks_like_chat(question):
        intent = "web"
    return {
        "intent": intent,
        "route_reason": f"LLM intent classifier → {intent}",
    }


def _route_after_understand(
    state: ChatState,
) -> Literal["chat_reply", "rewrite"]:
    if (state.get("intent") or "") == "chat":
        return "chat_reply"
    return "rewrite"


def _chat_reply(state: ChatState) -> dict[str, Any]:
    """Direct LLM reply — no document retrieval, no web search."""
    llm = get_llm(temperature=0.4, reasoning=False)
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a friendly assistant in a document Q&A lab. "
                    "Respond naturally to greetings and small talk. "
                    "If the user asks whether you can search the internet / web, "
                    "say YES — you can search when they turn on Search or ask a concrete "
                    "question like 'search the web for …'. Invite them to give a topic. "
                    "If they ask whether you can read documents, say YES — they can upload "
                    "a file and ask about it. "
                    "Do NOT invent contents of uploaded documents. "
                    "Do NOT pretend you already searched the web or opened a file. "
                    "Keep it brief (1–3 sentences)."
                )
            ),
            HumanMessage(content=state["question"]),
        ]
    )
    return {
        "answer": str(response.content).strip(),
        "verified": True,
        "evidence_grade": "n/a",
        "context": "",
        "web_context": "",
        "sources": [],
        "web_results": [],
        "chunk_candidates": [],
        "chunk_previews": [],
        "rerank_backend": "",
        "doc_score": 0.0,
        "web_score": 0.0,
    }


def _rewrite(state: ChatState) -> dict[str, Any]:
    llm = get_llm(temperature=0, reasoning=False)
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Rewrite the user question into a precise retrieval/search query. "
                    "Keep named entities, dates, and product names. "
                    "Return only the query text."
                )
            ),
            HumanMessage(content=state["question"]),
        ]
    )
    rewritten = str(response.content).strip().strip('"') or state["question"]
    queries = build_search_queries(state["question"], rewritten)
    return {"rewritten_query": rewritten, "search_queries": queries}


def _route_after_rewrite(
    state: ChatState,
) -> Literal["retrieve", "web_search"]:
    intent = state.get("intent") or "documents"
    if intent == "web":
        return "web_search"
    return "retrieve"


def _retrieve(state: ChatState) -> dict[str, Any]:
    """Broad similarity search — keep many candidates for the rerank node."""
    store = get_store()
    query = state.get("rewritten_query") or state["question"]
    hits = store.retrieve(
        state["workspace_id"],
        query,
        k=retrieve_candidate_count(),
    )
    candidates: list[dict[str, Any]] = [
        {
            "source": hit.filename,
            "content": hit.content,
            "score": float(hit.score),
            "chunk_id": hit.chunk_id,
        }
        for hit in hits
    ]
    # Context / doc_score are finalized in `_rerank` after re-scoring.
    return {
        "chunk_candidates": candidates,
        "context": "",
        "sources": [],
        "chunk_previews": [],
        "doc_score": 0.0,
        "rerank_backend": "",
    }


def _rerank(state: ChatState) -> dict[str, Any]:
    """Re-score retrieve candidates and keep the best top-k for generation."""
    query = state.get("rewritten_query") or state["question"]
    question = state["question"]
    candidates = list(state.get("chunk_candidates") or [])
    ranked, backend = rerank_chunks(query, candidates)
    context, sources, previews, doc_score = build_context_from_ranked(question, ranked)
    # Prefer question tokens for overlap; rewritten query already drove retrieval order.
    return {
        "context": context,
        "sources": sources,
        "chunk_previews": previews,
        "doc_score": doc_score,
        "rerank_backend": backend,
        # Drop bulky candidates from later state snapshots.
        "chunk_candidates": [],
    }


def _route_after_rerank(state: ChatState) -> Literal["web_search", "grade"]:
    intent = state.get("intent") or "documents"
    # hybrid always searches; documents + Search toggle also searches.
    if intent == "hybrid" or state.get("use_web_search"):
        return "web_search"
    return "grade"


def _web_search(state: ChatState) -> dict[str, Any]:
    queries = state.get("search_queries") or [
        state.get("rewritten_query") or state["question"]
    ]
    hits = search_web(
        queries[0],
        max_results=6,
        extra_queries=queries[1:],
        retries=2,
    )
    usable = [h for h in hits if h.get("url")]
    web_score = 0.0
    if usable:
        web_score = max(
            evidence_overlap(state["question"], f"{h['title']} {h['snippet']}")
            for h in usable
        )
    return {
        "web_results": usable,
        "web_context": format_web_context(usable),
        "web_score": round(web_score, 4),
    }


def _grade(state: ChatState) -> dict[str, Any]:
    doc_score = float(state.get("doc_score") or 0)
    web_score = float(state.get("web_score") or 0)
    has_docs = bool((state.get("context") or "").strip())
    has_web = bool((state.get("web_context") or "").strip())
    if has_docs and doc_score >= 0.18:
        grade = "pass"
    elif has_web and web_score >= 0.12:
        grade = "pass"
    elif has_docs or has_web:
        grade = "weak"
    else:
        grade = "fail"
    return {"evidence_grade": grade}


def _generate(state: ChatState) -> dict[str, Any]:
    doc_context = state.get("context") or ""
    web_context = state.get("web_context") or ""
    grade = state.get("evidence_grade") or "fail"
    intent = state.get("intent") or "documents"

    if grade == "fail" or (not doc_context and not web_context):
        hint = (
            "Turn on Search for live web facts, or ask about an uploaded document."
            if intent != "web"
            else "Try a more specific question, or check your network / search provider."
        )
        return {
            "answer": (
                "I couldn't find reliable evidence for that. " + hint
            ),
            "verified": False,
        }

    sections: list[str] = []
    if doc_context:
        sections.append(f"Uploaded document excerpts:\n{doc_context}")
    if web_context:
        sections.append(f"Live web results:\n{web_context}")
    combined = "\n\n".join(sections)

    prefer = "documents" if (state.get("doc_score") or 0) >= (state.get("web_score") or 0) else "web"
    system = (
        "You are a careful research assistant. "
        "ONLY use facts present in the provided excerpts. "
        "Do NOT invent dates, scores, winners, prices, or URLs. "
        f"Prefer {prefer} evidence when both exist. "
        "Cite documents as [exact-filename] and web results as [W1], [W2]. "
        "If evidence is weak or conflicting, say what is uncertain. "
        "If excerpts do not answer the question, say you cannot verify it from the sources."
    )
    if grade == "weak":
        system += " Evidence is weak — be especially cautious and hedge clearly."

    llm = get_llm(temperature=0, reasoning=False)
    response = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(
                content=(
                    f"Question: {state['question']}\n\n"
                    f"Intent route: {intent}\n"
                    f"Evidence grade: {grade}\n"
                    f"Doc relevance: {state.get('doc_score')}\n"
                    f"Web relevance: {state.get('web_score')}\n\n"
                    f"{combined}"
                )
            ),
        ]
    )
    return {"answer": str(response.content).strip()}


def _token_support(answer: str, evidence: str) -> float:
    tokens = [
        t
        for t in re.findall(r"[a-z0-9]{4,}", answer.lower())
        if t
        not in {
            "that",
            "this",
            "with",
            "from",
            "have",
            "were",
            "been",
            "also",
            "into",
            "your",
            "their",
            "about",
            "which",
            "would",
            "could",
            "should",
            "there",
            "source",
            "according",
            "however",
        }
    ]
    if not tokens:
        return 1.0
    blob = evidence.lower()
    supported = sum(1 for t in tokens[:40] if t in blob)
    return supported / max(min(len(tokens), 40), 1)


def _verify(state: ChatState) -> dict[str, Any]:
    answer = state.get("answer") or ""
    sources = state.get("sources") or []
    web_results = state.get("web_results") or []
    evidence = f"{state.get('context') or ''}\n{state.get('web_context') or ''}"
    if not evidence.strip():
        return {"verified": False}

    lower = answer.lower()
    refusal = any(
        phrase in lower
        for phrase in (
            "cannot verify",
            "can't verify",
            "could not find",
            "couldn't find",
            "no reliable",
            "don't have",
            "do not have",
            "not enough evidence",
            "unable to find",
        )
    )
    has_doc_cite = any(name in answer for name in sources) if sources else False
    has_web_cite = any(
        f"[W{i}]" in answer or f"[W{i} " in answer for i in range(1, len(web_results) + 1)
    )
    support = _token_support(answer, evidence)

    claimed_urls = re.findall(r"https?://[^\s)\]]+", answer)
    allowed = {h.get("url") for h in web_results if h.get("url")}
    bad_urls = [u for u in claimed_urls if u not in allowed]

    verified = False
    if refusal:
        verified = True
    elif bad_urls:
        verified = False
    elif (has_doc_cite or has_web_cite) and support >= 0.35:
        verified = True
    elif support >= 0.55 and (sources or web_results):
        verified = True

    return {"verified": verified}


def _route_after_verify(state: ChatState) -> Literal["fix", "__end__"]:
    if state.get("verified"):
        return "__end__"
    if int(state.get("fix_attempts") or 0) >= 1:
        return "__end__"
    return "fix"


def _fix(state: ChatState) -> dict[str, Any]:
    llm = get_llm(temperature=0, reasoning=False)
    evidence = f"{state.get('context') or ''}\n\n{state.get('web_context') or ''}".strip()
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "The previous answer failed verification (possible hallucination). "
                    "Rewrite using ONLY the evidence below. "
                    "If you cannot support the answer, say you cannot verify it from the sources. "
                    "Cite [filename] or [W#] when using evidence."
                )
            ),
            HumanMessage(
                content=(
                    f"Question: {state['question']}\n\n"
                    f"Previous answer:\n{state.get('answer')}\n\n"
                    f"Evidence:\n{evidence or '(none)'}"
                )
            ),
        ]
    )
    return {
        "answer": str(response.content).strip(),
        "fix_attempts": int(state.get("fix_attempts") or 0) + 1,
        "verified": False,
    }


def build_graph():
    graph = StateGraph(ChatState)
    graph.add_node("understand", _understand)
    graph.add_node("chat_reply", _chat_reply)
    graph.add_node("rewrite", _rewrite)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("rerank", _rerank)
    graph.add_node("web_search", _web_search)
    graph.add_node("grade", _grade)
    graph.add_node("generate", _generate)
    graph.add_node("verify", _verify)
    graph.add_node("fix", _fix)

    graph.add_edge(START, "understand")
    graph.add_conditional_edges(
        "understand",
        _route_after_understand,
        {"chat_reply": "chat_reply", "rewrite": "rewrite"},
    )
    graph.add_edge("chat_reply", END)
    graph.add_conditional_edges(
        "rewrite",
        _route_after_rewrite,
        {"retrieve": "retrieve", "web_search": "web_search"},
    )
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges(
        "rerank",
        _route_after_rerank,
        {"web_search": "web_search", "grade": "grade"},
    )
    graph.add_edge("web_search", "grade")
    graph.add_edge("grade", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify",
        _route_after_verify,
        {"fix": "fix", "__end__": END},
    )
    graph.add_edge("fix", "verify")
    return graph.compile()


_GRAPH = None


def reset_graph() -> None:
    global _GRAPH
    _GRAPH = None


def _empty_state(workspace_id: str, question: str, web_search: bool) -> dict[str, Any]:
    return {
        "workspace_id": workspace_id,
        "question": question,
        "use_web_search": web_search,
        "intent": "",
        "route_reason": "",
        "rewritten_query": "",
        "search_queries": [],
        "context": "",
        "web_context": "",
        "sources": [],
        "web_results": [],
        "chunk_candidates": [],
        "chunk_previews": [],
        "rerank_backend": "",
        "doc_score": 0.0,
        "web_score": 0.0,
        "evidence_grade": "fail",
        "answer": "",
        "verified": False,
        "fix_attempts": 0,
    }


def ask(
    workspace_id: str,
    question: str,
    *,
    web_search: bool = False,
) -> dict[str, Any]:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    result = _GRAPH.invoke(_empty_state(workspace_id, question, web_search))
    store = get_store()
    intent = result.get("intent") or ""
    return {
        "answer": result["answer"],
        "intent": intent,
        "route_reason": result.get("route_reason") or "",
        "sources": result.get("sources") or [],
        "web_results": result.get("web_results") or [],
        "chunk_previews": result.get("chunk_previews") or [],
        "rewritten_query": result.get("rewritten_query") or question,
        "search_queries": result.get("search_queries") or [],
        "doc_score": result.get("doc_score"),
        "web_score": result.get("web_score"),
        "evidence_grade": result.get("evidence_grade"),
        "rerank_backend": result.get("rerank_backend") or "",
        "verified": bool(result.get("verified")),
        "web_search_used": bool(result.get("web_results")),
        "workspace": {
            "id": workspace_id,
            **store.stats(workspace_id),
        },
    }
