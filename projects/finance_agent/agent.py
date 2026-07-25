"""F7 LangGraph L1–L2 finance analysis agent.

Read-only tools only. The graph cannot place, propose, or approve orders.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from Learning.llm import get_llm
from projects.finance_agent import db
from projects.finance_agent.agent_tools import (
    corp_actions_tool,
    filings_tool,
    fundamentals_tool,
    quote_tool,
)
from projects.finance_agent.config import FinanceConfig, load_config

Intent = Literal[
    "quote",
    "fundamentals",
    "corp_actions",
    "filings",
    "brief",
    "rules",
]

_ORDER_RE = re.compile(
    r"\b(buy|sell|place|execute|submit|approve)\b.{0,24}\b(order|trade|shares?|stock)\b"
    r"|\b(order|trade)\b.{0,24}\b(buy|sell|execute|place)\b"
    r"|\bshould\s+i\s+(buy|sell)\b"
    r"|\b(recommend|tell\s+me)\b.{0,24}\b(buy|sell)\b"
    r"|\b(entry|exit)\s+(price|point)\b|\bprice\s+target\b",
    re.IGNORECASE,
)
_FUNDAMENTAL_RE = re.compile(
    r"\b(revenue|ebitda|profit|pat|debt|cash|asset|equity|balance sheet|"
    r"income statement|cash flow|fundamental|annual|quarter)\b",
    re.IGNORECASE,
)
_FILING_RE = re.compile(
    r"\b(filing|annual report|announcement|management|outlook|risk|said|"
    r"commentary|disclosure)\b",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"\b(dividend|bonus|split|buyback|corporate action|record date|ex-date)\b",
    re.IGNORECASE,
)
_QUOTE_RE = re.compile(
    r"\b(price|quote|ltp|volume|bid|ask|market)\b",
    re.IGNORECASE,
)


class AgentState(TypedDict):
    question: str
    symbol: str
    exchange: str
    level: str
    intent: str
    route_reason: str
    plan: list[str]
    tool_trace: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]
    verified: bool
    refused: bool


def _clean(text: object) -> str:
    raw = text if isinstance(text, str) else str(text)
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def _known_symbols(config: FinanceConfig) -> set[str]:
    with db.connect(config) as conn:
        rows = conn.execute(
            "SELECT DISTINCT symbol FROM finance_company_master"
        ).fetchall()
    return {str(row[0]).upper() for row in rows}


def resolve_symbol(
    question: str,
    symbol: str | None = None,
    *,
    config: FinanceConfig | None = None,
) -> str:
    if symbol and symbol.strip():
        return symbol.strip().upper()
    cfg = config or load_config()
    known = _known_symbols(cfg)
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9.&-]{1,31}\b", question)
    for word in words:
        candidate = word.upper()
        if candidate in known:
            return candidate
    raise ValueError("symbol is required (for example: RELIANCE, TCS, or INFY)")


def _route(state: AgentState) -> dict[str, Any]:
    question = state["question"]
    if _ORDER_RE.search(question):
        return {
            "intent": "rules",
            "route_reason": "Trading/order request blocked by the analysis-only safety policy.",
            "plan": ["enforce_analysis_only_policy"],
            "refused": True,
        }
    if state.get("level") == "L1":
        intent = "corp_actions" if _ACTION_RE.search(question) else "quote"
        return {
            "intent": intent,
            "route_reason": (
                "L1 Observe is limited to read-only market observations "
                "and corporate-action facts."
            ),
            "plan": ["get_corp_actions"] if intent == "corp_actions" else ["get_quote"],
            "refused": False,
        }
    domains = {
        "corp_actions": bool(_ACTION_RE.search(question)),
        "filings": bool(_FILING_RE.search(question)),
        "fundamentals": bool(_FUNDAMENTAL_RE.search(question)),
        "quote": bool(_QUOTE_RE.search(question)),
    }
    intent: Intent
    if sum(domains.values()) > 1:
        intent = "brief"
        reason = "Multiple evidence domains detected; gather a multi-source brief."
    elif domains["corp_actions"]:
        intent = "corp_actions"
        reason = "Corporate-action terms detected."
    elif domains["filings"]:
        intent = "filings"
        reason = "Narrative filing/research terms detected."
    elif domains["fundamentals"]:
        intent = "fundamentals"
        reason = "Financial-statement terms detected."
    elif domains["quote"]:
        intent = "quote"
        reason = "Market quote terms detected."
    else:
        intent = "brief"
        reason = "General company analysis request; gather a compact multi-source brief."

    plans = {
        "quote": ["get_quote"],
        "fundamentals": ["get_fundamentals"],
        "corp_actions": ["get_corp_actions"],
        "filings": ["search_filings"],
        "brief": [
            "get_quote",
            "get_fundamentals",
            "get_corp_actions",
            "search_filings",
        ],
        "rules": ["enforce_analysis_only_policy"],
    }
    return {
        "intent": intent,
        "route_reason": reason,
        "plan": plans[intent],
        "refused": False,
    }


def _route_after_intent(state: AgentState) -> Literal["policy", "tools"]:
    return "policy" if state.get("intent") == "rules" else "tools"


def _policy(state: AgentState) -> dict[str, Any]:
    return {
        "answer": (
            "I can analyse quotes, fundamentals, corporate actions, and filings, "
            "but I cannot place, propose, or approve buy/sell orders. "
            "Ask for a cited research brief instead."
        ),
        "citations": [],
        "verified": True,
        "tool_trace": [
            {
                "tool": "analysis_only_policy",
                "status": "blocked",
                "detail": "No brokerage or order tools exist in this graph.",
            }
        ],
    }


def _latest_fundamental_lines(data: dict[str, Any]) -> list[dict[str, Any]]:
    periods = data.get("available_periods") or []
    selected = periods[:2]
    keys = {f"{p['period_type']}:{p['period']}" for p in selected}
    return [
        line
        for line in data.get("lines") or []
        if f"{line['period_type']}:{line['period']}" in keys
    ][:18]


def _tools(state: AgentState) -> dict[str, Any]:
    symbol = state["symbol"]
    exchange = state["exchange"]
    question = state["question"]
    plan = state.get("plan") or []
    evidence: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []

    for tool_name in plan:
        try:
            if tool_name == "get_quote":
                data = quote_tool(symbol, exchange)
                if data:
                    evidence.append(
                        {
                            "id": "Q1",
                            "kind": "quote",
                            "source": data.get("source") or "quote feed",
                            "data": data,
                        }
                    )
                count = 1 if data else 0
            elif tool_name == "get_fundamentals":
                raw = fundamentals_tool(symbol, exchange)
                lines = _latest_fundamental_lines(raw)
                if lines:
                    evidence.append(
                        {
                            "id": "F1",
                            "kind": "fundamentals",
                            "source": lines[0].get("source") or "fundamentals",
                            "data": lines,
                        }
                    )
                count = len(lines)
            elif tool_name == "get_corp_actions":
                actions = corp_actions_tool(symbol)
                if actions:
                    evidence.append(
                        {
                            "id": "A1",
                            "kind": "corporate_actions",
                            "source": actions[0].get("source") or "corporate actions",
                            "data": actions,
                        }
                    )
                count = len(actions)
            elif tool_name == "search_filings":
                result = filings_tool(symbol, question)
                chunks = result.get("chunks") or []
                for index, chunk in enumerate(chunks, start=1):
                    evidence.append(
                        {
                            "id": f"D{index}",
                            "kind": "filing",
                            "source": chunk.get("source") or "filing",
                            "title": chunk.get("title"),
                            "data": chunk.get("content") or "",
                            "score": chunk.get("rerank_score"),
                        }
                    )
                count = len(chunks)
            else:
                continue
            trace.append({"tool": tool_name, "status": "ok", "items": count})
        except Exception as exc:  # noqa: BLE001
            trace.append(
                {"tool": tool_name, "status": "error", "detail": str(exc)}
            )

    return {"evidence": evidence, "tool_trace": trace}


def _format_evidence(evidence: list[dict[str, Any]]) -> str:
    blocks = []
    for item in evidence:
        blocks.append(
            f"[{item['id']}] kind={item['kind']} source={item['source']}\n"
            f"{json.dumps(item.get('data'), ensure_ascii=False, default=str)}"
        )
    return "\n\n".join(blocks)


def _fallback_answer(state: AgentState) -> str:
    evidence = state.get("evidence") or []
    if not evidence:
        return (
            "I could not find local evidence for this request. Ingest quotes, "
            "fundamentals, and filings, then try again."
        )
    lines = [f"Analysis brief for {state['symbol']}:"]
    for item in evidence[:5]:
        if item["kind"] == "quote":
            data = item["data"]
            lines.append(
                f"- Latest local quote: {data.get('ltp')} "
                f"({data.get('source')}). [{item['id']}]"
            )
        elif item["kind"] == "fundamentals":
            sample = item["data"][:4]
            summary = ", ".join(
                f"{row['line_item']}={row['value']} {row.get('unit') or row.get('currency')}"
                f" ({row['period']})"
                for row in sample
            )
            lines.append(f"- Statement snapshot: {summary}. [{item['id']}]")
        elif item["kind"] == "corporate_actions":
            action = item["data"][0]
            lines.append(
                f"- Latest corporate action: {action['action_type']} "
                f"(ex-date {action.get('ex_date') or 'not supplied'}). [{item['id']}]"
            )
        elif item["kind"] == "filing":
            text = re.sub(r"\s+", " ", str(item["data"])).strip()[:260]
            lines.append(f"- Filing evidence: {text} [{item['id']}]")
    lines.append(
        "\nAnalysis only. This is not investment advice and no order action is available."
    )
    return "\n".join(lines)


def _generate(state: AgentState) -> dict[str, Any]:
    evidence = state.get("evidence") or []
    if not evidence:
        return {"answer": _fallback_answer(state)}
    system = (
        f"You are an India-first markets research assistant operating at "
        f"{state['level']} {'Observe' if state['level'] == 'L1' else 'Analyze'}. "
        "Use ONLY the supplied evidence. Write a concise answer with a short conclusion, "
        "supporting facts, and risks/uncertainties where relevant. Cite every factual "
        "claim with the exact evidence ID, such as [Q1], [F1], or [D1]. "
        "Never give a buy/sell recommendation, price target, order instruction, or "
        "personalised investment advice. Never invent missing dates or values. "
        "End with: 'Analysis only — not investment advice.'"
    )
    try:
        response = get_llm(temperature=0.1, reasoning=False).invoke(
            [
                SystemMessage(content=system),
                HumanMessage(
                    content=(
                        f"Symbol: {state['symbol']} ({state['exchange']})\n"
                        f"Question: {state['question']}\n"
                        f"Intent: {state['intent']}\n\n"
                        f"EVIDENCE:\n{_format_evidence(evidence)}"
                    )
                ),
            ]
        )
        answer = _clean(response.content)
    except Exception:  # noqa: BLE001
        answer = _fallback_answer(state)
    return {"answer": answer}


def _cite_and_verify(state: AgentState) -> dict[str, Any]:
    answer = state.get("answer") or ""
    evidence = state.get("evidence") or []
    allowed = {item["id"] for item in evidence}
    cited = set(re.findall(r"\[([QFAD]\d+)\]", answer))
    citations = [
        {
            "id": item["id"],
            "kind": item["kind"],
            "source": item["source"],
            "title": item.get("title"),
            "score": item.get("score"),
        }
        for item in evidence
        if item["id"] in cited
    ]
    has_unsupported_cite = bool(cited - allowed)
    has_evidence_cite = bool(cited & allowed)
    unsafe = bool(_ORDER_RE.search(answer))
    verified = (has_evidence_cite or not evidence) and not has_unsupported_cite and not unsafe
    if evidence and not has_evidence_cite:
        answer = _fallback_answer(state)
        cited = set(re.findall(r"\[([QFAD]\d+)\]", answer))
        citations = [
            {
                "id": item["id"],
                "kind": item["kind"],
                "source": item["source"],
                "title": item.get("title"),
                "score": item.get("score"),
            }
            for item in evidence
            if item["id"] in cited
        ]
        verified = bool(citations)
    return {
        "answer": answer,
        "citations": citations,
        "verified": verified,
    }


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("route_intent", _route)
    graph.add_node("policy", _policy)
    graph.add_node("tools", _tools)
    graph.add_node("generate", _generate)
    graph.add_node("cite_sources", _cite_and_verify)
    graph.add_edge(START, "route_intent")
    graph.add_conditional_edges(
        "route_intent",
        _route_after_intent,
        {"policy": "policy", "tools": "tools"},
    )
    graph.add_edge("policy", END)
    graph.add_edge("tools", "generate")
    graph.add_edge("generate", "cite_sources")
    graph.add_edge("cite_sources", END)
    return graph.compile()


_GRAPH = None


def analyse(
    question: str,
    *,
    symbol: str | None = None,
    exchange: str = "NSE",
    level: str = "L2",
    config: FinanceConfig | None = None,
) -> dict[str, Any]:
    global _GRAPH
    q = (question or "").strip()
    if not q:
        raise ValueError("question is required")
    normalized_level = level.upper()
    if normalized_level not in {"L1", "L2"}:
        raise ValueError("level must be L1 or L2")
    cfg = config or load_config()
    resolved = resolve_symbol(q, symbol, config=cfg)
    if _GRAPH is None:
        _GRAPH = build_graph()
    result = _GRAPH.invoke(
        {
            "question": q,
            "symbol": resolved,
            "exchange": exchange.upper(),
            "level": normalized_level,
            "intent": "",
            "route_reason": "",
            "plan": [],
            "tool_trace": [],
            "evidence": [],
            "answer": "",
            "citations": [],
            "verified": False,
            "refused": False,
        }
    )
    return {
        "symbol": resolved,
        "exchange": exchange.upper(),
        "level": normalized_level,
        "intent": result.get("intent"),
        "route_reason": result.get("route_reason"),
        "plan": result.get("plan") or [],
        "tool_trace": result.get("tool_trace") or [],
        "answer": result.get("answer") or "",
        "citations": result.get("citations") or [],
        "verified": bool(result.get("verified")),
        "refused": bool(result.get("refused")),
        "analysis_only": True,
        "orders_supported": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finance F7 L1-L2 analysis agent")
    parser.add_argument("question")
    parser.add_argument("--symbol")
    parser.add_argument("--exchange", default="NSE")
    parser.add_argument("--level", default="L2", choices=["L1", "L2"])
    args = parser.parse_args(argv)
    print(
        json.dumps(
            analyse(
                args.question,
                symbol=args.symbol,
                exchange=args.exchange,
                level=args.level,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
