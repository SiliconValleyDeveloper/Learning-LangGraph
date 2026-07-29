"""Multi-agent shipping graph with human approval before business writes."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from Learning.llm import get_llm
from projects.shipping_logistics_agent import repository
from projects.shipping_logistics_agent.config import load_config
from projects.shipping_logistics_agent.prompt_parser import (
    classify_action,
    extract_parameters,
)
from projects.shipping_logistics_agent.rag import (
    build_candidates,
    evidence_context,
    grade_evidence,
    policy_candidates,
    rerank_evidence,
    verify_answer,
)

WRITE_ACTIONS = {"create_quotation", "create_booking"}


class ShippingState(TypedDict):
    thread_id: str
    prompt: str
    action: str
    plan: list[str]
    parameters: dict[str, Any]
    proposal: dict[str, Any]
    risk_review: dict[str, Any]
    approval: dict[str, Any]
    result: dict[str, Any] | list[dict[str, Any]]
    response: dict[str, Any]
    status: str
    errors: list[str]
    trace: list[dict[str, Any]]
    lane: str
    route_reason: str
    rewritten_query: str
    search_queries: list[str]
    candidates: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    evidence_grade: str
    evidence_score: float
    rerank_backend: str
    retrieval_attempts: int
    answer: str
    verified: bool
    verification_issues: list[str]
    fix_attempts: int


GRAPH_NODES = [
    {"id": "__start__", "label": "START", "kind": "start"},
    {"id": "understand", "label": "Understand intent", "kind": "router"},
    {"id": "rewrite", "label": "Rewrite query", "kind": "agent"},
    {"id": "retrieve", "label": "Retrieve evidence", "kind": "tools"},
    {"id": "rerank", "label": "Rerank evidence", "kind": "agent"},
    {"id": "grade", "label": "Grade evidence", "kind": "router"},
    {"id": "generate", "label": "Generate answer", "kind": "agent"},
    {"id": "verify", "label": "Verify answer", "kind": "router"},
    {"id": "fix", "label": "Fix answer", "kind": "agent"},
    {"id": "operations", "label": "Operations agent", "kind": "agent"},
    {"id": "pricing", "label": "Pricing agent", "kind": "agent"},
    {"id": "risk", "label": "Risk/compliance agent", "kind": "agent"},
    {"id": "approval_request", "label": "Approval request", "kind": "approval"},
    {"id": "human", "label": "HUMAN APPROVAL", "kind": "interrupt"},
    {"id": "execute", "label": "Approved PostgreSQL write", "kind": "tools"},
    {"id": "response", "label": "JSON response", "kind": "agent"},
    {"id": "__end__", "label": "END", "kind": "end"},
]

GRAPH_EDGES = [
    {"source": "__start__", "target": "understand", "label": ""},
    {"source": "understand", "target": "response", "label": "chat"},
    {"source": "understand", "target": "rewrite", "label": "read / RAG"},
    {"source": "understand", "target": "operations", "label": "write"},
    {"source": "rewrite", "target": "retrieve", "label": "query"},
    {"source": "retrieve", "target": "rerank", "label": "candidates"},
    {"source": "rerank", "target": "grade", "label": "top-k"},
    {"source": "grade", "target": "rewrite", "label": "retry"},
    {"source": "grade", "target": "generate", "label": "pass / give up"},
    {"source": "generate", "target": "verify", "label": "draft"},
    {"source": "verify", "target": "fix", "label": "unsupported"},
    {"source": "fix", "target": "verify", "label": "recheck"},
    {"source": "verify", "target": "response", "label": "verified"},
    {"source": "operations", "target": "response", "label": "invalid"},
    {"source": "operations", "target": "pricing", "label": "quote/booking"},
    {"source": "pricing", "target": "risk", "label": "proposal"},
    {"source": "pricing", "target": "response", "label": "invalid proposal"},
    {"source": "risk", "target": "response", "label": "hard block"},
    {"source": "risk", "target": "approval_request", "label": "reviewable"},
    {"source": "approval_request", "target": "human", "label": "pause"},
    {"source": "human", "target": "execute", "label": "approve/reject"},
    {"source": "execute", "target": "response", "label": "result"},
    {"source": "response", "target": "__end__", "label": ""},
]

MERMAID = """flowchart TD
    START([START]) --> UND{Understand intent}
    UND -->|chat| JSON[JSON response]
    UND -->|read / RAG| RW[Rewrite query]
    RW --> RET[Retrieve SQL + policy evidence]
    RET --> RR[Rerank top-k]
    RR --> GR{Grade evidence}
    GR -->|retry| RW
    GR -->|pass / give up| GEN[Generate grounded answer]
    GEN --> VER{Verify citations and references}
    VER -->|fix| FIX[Fix answer]
    FIX --> VER
    VER -->|verified| JSON
    UND -->|write| OPS[Operations agent]
    OPS -->|quotation / booking| PRICE[Pricing agent]
    OPS -->|invalid| JSON
    PRICE --> RISK[Risk & compliance]
    RISK -->|hard block| JSON
    RISK -->|reviewable| REQ[Approval request]
    REQ --> HUMAN{{Human approval}}
    HUMAN -->|approve / reject| EXEC[Approved PostgreSQL write]
    EXEC --> JSON
    JSON --> END([END])
"""


def _trace(
    state: ShippingState,
    agent: str,
    summary: str,
    **extra: Any,
) -> list[dict[str, Any]]:
    entries = list(state.get("trace") or [])
    entries.append({"agent": agent, "summary": summary, **extra})
    return entries


def supervisor_agent(state: ShippingState) -> dict[str, Any]:
    action = classify_action(state["prompt"])
    if action == "conversation":
        lane = "chat"
    elif action in WRITE_ACTIONS:
        lane = "write"
    else:
        lane = "rag"
    rag_plan = [
        "rewrite_agent",
        "retrieve_agent",
        "rerank_agent",
        "grade_agent",
        "generate_agent",
        "verify_agent",
        "response_agent",
    ]
    plans = {
        "conversation": ["response_agent"],
        "reference_data": rag_plan,
        "data_query": rag_plan,
        "get_quotation": rag_plan,
        "search_sailings": rag_plan,
        "track_booking": rag_plan,
        "create_quotation": [
            "operations_agent",
            "pricing_agent",
            "risk_agent",
            "human_approval",
            "execute",
        ],
        "create_booking": [
            "operations_agent",
            "pricing_agent",
            "risk_agent",
            "human_approval",
            "execute",
        ],
    }
    return {
        "action": action,
        "plan": plans[action],
        "lane": lane,
        "route_reason": f"{action} uses the {lane} lane",
        "status": "planned",
        "trace": _trace(
            state,
            "understand_agent",
            f"Understood {action}; routed to {lane} lane",
            action=action,
            lane=lane,
        ),
    }


def _after_understand(
    state: ShippingState,
) -> Literal["response", "rewrite", "operations"]:
    lane = state.get("lane")
    if lane == "chat":
        return "response"
    if lane == "write":
        return "operations"
    return "rewrite"


def operations_agent(state: ShippingState) -> dict[str, Any]:
    action = state["action"]
    params = (
        {}
        if action == "conversation"
        else extract_parameters(state["prompt"], action)
    )
    errors: list[str] = []
    result: dict[str, Any] | list[dict[str, Any]] = {}
    for role in ("origin", "destination"):
        unknown = params.get(f"unrecognized_{role}")
        if unknown:
            errors.append(
                f"Unsupported {role} port code {unknown}"
            )

    try:
        if action == "conversation":
            result = {
                "capabilities": [
                    "search sailings",
                    "prepare quotations",
                    "create approved bookings",
                    "track shipments",
                ]
            }
        elif action == "reference_data":
            result = repository.list_reference_data()
        elif action == "data_query":
            if not params.get("entity"):
                errors.append(
                    "Missing data query parameters: entity "
                    "(customers, ports, vessels, sailings, quotations, "
                    "bookings, containers, or shipment events)"
                )
            elif params.get("entity") == "all":
                result = repository.shipping_data_overview()
            else:
                result = repository.query_shipping_data(
                    params["entity"],
                    operation=params.get("query_mode") or "list",
                    status=params.get("status"),
                    customer_code=params.get("customer_code"),
                    origin=params.get("origin"),
                    destination=params.get("destination"),
                    reference=params.get("reference"),
                    limit=int(params.get("limit") or 10),
                )
        elif action == "get_quotation":
            result = repository.get_quotation(params["quote_ref"])
            if not result:
                errors.append("Quotation not found")
        elif action == "search_sailings":
            missing = [
                key for key in ("origin", "destination") if not params.get(key)
            ]
            if missing:
                available = repository.query_shipping_data(
                    "sailings",
                    operation="list",
                    limit=10,
                )
                ports = repository.list_reference_data()["ports"]
                result = {
                    "matched": [],
                    "message": (
                        "Route origin and destination are required for a sailing "
                        "search. Available sample ports and sailings are included."
                    ),
                    "available_ports": ports,
                    "available_sailings": available.get("records") or [],
                }
                # Keep asking for the route, but still return useful evidence.
                errors.append(
                    "Missing route parameters: " + ", ".join(missing)
                )
            else:
                matched = repository.search_sailings(
                    params["origin"],
                    params["destination"],
                    departure_after=params.get("departure_after"),
                )
                if matched:
                    result = matched
                else:
                    available = repository.query_shipping_data(
                        "sailings",
                        operation="list",
                        limit=10,
                    )
                    ports = repository.list_reference_data()["ports"]
                    result = {
                        "matched": [],
                        "origin": params["origin"],
                        "destination": params["destination"],
                        "message": (
                            f"No scheduled sailing from {params['origin']} to "
                            f"{params['destination']} was found."
                        ),
                        "available_ports": ports,
                        "available_sailings": available.get("records") or [],
                    }
        elif action == "track_booking":
            if not params.get("booking_ref"):
                errors.append("Missing booking reference (SLB-...)")
            else:
                result = repository.track_booking(params["booking_ref"])
                if not result:
                    errors.append("Booking not found")
        elif action == "create_quotation":
            references = repository.list_reference_data()
            known_customers = {
                customer["customer_code"]
                for customer in references.get("customers") or []
            }
            customer_code = params.get("customer_code")
            if customer_code and customer_code not in known_customers:
                errors.append(f"Unknown customer_code {customer_code}")

            sailing_catalog = (
                repository.query_shipping_data(
                    "sailings",
                    operation="list",
                    limit=25,
                ).get("records")
                or []
            )
            if not params.get("sailing_id") and params.get("voyage_number"):
                matched = [
                    sailing
                    for sailing in sailing_catalog
                    if sailing.get("voyage_number") == params["voyage_number"]
                ]
                if matched:
                    selected = matched[0]
                    route_matches = (
                        not params.get("origin")
                        or selected.get("origin") == params["origin"]
                    ) and (
                        not params.get("destination")
                        or selected.get("destination") == params["destination"]
                    )
                    if not route_matches:
                        errors.append(
                            f"Voyage {params['voyage_number']} does not match "
                            f"{params.get('origin')} to {params.get('destination')}"
                        )
                    elif selected.get("status") != "scheduled":
                        errors.append(
                            f"Voyage {params['voyage_number']} is not scheduled"
                        )
                    else:
                        params["sailing_id"] = selected["sailing_id"]
                        params["selected_sailing"] = selected
                else:
                    errors.append(
                        f"Voyage {params['voyage_number']} was not found"
                    )
            elif params.get("sailing_id"):
                selected = next(
                    (
                        sailing
                        for sailing in sailing_catalog
                        if sailing.get("sailing_id") == params["sailing_id"]
                    ),
                    None,
                )
                if not selected:
                    errors.append(
                        f"Unknown sailing_id {params['sailing_id']}"
                    )
                elif selected.get("status") != "scheduled":
                    errors.append(
                        f"Sailing {params['sailing_id']} is not scheduled"
                    )
                else:
                    params["voyage_number"] = selected["voyage_number"]
                    params["selected_sailing"] = selected
            if not params.get("sailing_id") and params.get(
                "origin"
            ) and params.get("destination"):
                sailings = repository.search_sailings(
                    params["origin"],
                    params["destination"],
                    departure_after=params.get("departure_after"),
                    limit=1,
                )
                if sailings:
                    params["sailing_id"] = sailings[0]["sailing_id"]
                    params["selected_sailing"] = sailings[0]
            required = (
                "customer_code",
                "sailing_id",
                "container_type",
                "container_qty",
                "cargo_weight_kg",
            )
            missing = [key for key in required if not params.get(key)]
            if missing:
                errors.append(
                    "Missing quotation parameters: " + ", ".join(missing)
                )
        elif action == "create_booking":
            if not params.get("quote_ref"):
                errors.append("Missing quotation reference (SLQ-...)")
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))

    return {
        "parameters": params,
        "result": result,
        "errors": errors,
        "status": (
            "invalid_request"
            if errors
            else (
                "conversation_complete"
                if action == "conversation"
                else "operations_complete"
            )
        ),
        "trace": _trace(
            state,
            "operations_agent",
            "Resolved request parameters and queried PostgreSQL",
            parameters=params,
            errors=errors,
        ),
    }


def rewrite_agent(state: ShippingState) -> dict[str, Any]:
    original = state["prompt"]
    attempt = int(state.get("retrieval_attempts") or 0)
    instruction = (
        "Rewrite this shipping request into a precise retrieval query. Preserve "
        "every booking reference, quotation reference, voyage number, customer "
        "code, UN/LOCODE, status, date, and quantity exactly. Return only the query."
    )
    if attempt:
        instruction += (
            " The previous retrieval had insufficient evidence; use clearer shipping "
            "terms without changing identifiers."
        )
    try:
        if not load_config().use_llm_answers:
            raise RuntimeError("LLM rewriting disabled")
        response = get_llm(temperature=0, reasoning=False).invoke(
            [
                SystemMessage(content=instruction),
                HumanMessage(content=original),
            ]
        )
        rewritten = str(response.content).strip().strip('"') or original
    except Exception:  # noqa: BLE001
        rewritten = original
    queries = [original]
    if rewritten.lower() != original.lower():
        queries.append(rewritten)
    return {
        "rewritten_query": rewritten,
        "search_queries": queries,
        "status": "query_rewritten",
        "trace": _trace(
            state,
            "rewrite_agent",
            "Rewrote the request while preserving shipping identifiers",
            attempt=attempt,
            rewritten_query=rewritten,
        ),
    }


def retrieve_agent(state: ShippingState) -> dict[str, Any]:
    operation = operations_agent(state)
    errors = list(operation.get("errors") or [])
    result = operation.get("result") or {}
    # Always convert structured SQL results into evidence. Missing-route prompts
    # still include available sailings so the chatbot can explain sailing IDs.
    soft_errors = [
        error
        for error in errors
        if error.startswith("Missing route parameters:")
    ]
    hard_errors = [error for error in errors if error not in soft_errors]
    candidates = (
        []
        if hard_errors
        else build_candidates(state["action"], result)
    )
    if not hard_errors:
        candidates.extend(policy_candidates())
    attempt = int(state.get("retrieval_attempts") or 0) + 1
    return {
        "parameters": operation.get("parameters") or {},
        "result": result,
        "errors": errors,
        "candidates": candidates,
        "retrieval_attempts": attempt,
        "status": "evidence_retrieved" if candidates else "no_evidence",
        "trace": _trace(
            state,
            "retrieve_agent",
            f"Retrieved {len(candidates)} SQL and policy evidence candidates",
            attempt=attempt,
            candidate_count=len(candidates),
        ),
    }


def rerank_agent(state: ShippingState) -> dict[str, Any]:
    cfg = load_config()
    query = state.get("rewritten_query") or state["prompt"]
    top_k = cfg.rag_top_k
    # Keep full structured sailing catalogs so sailing_id answers stay complete.
    if state.get("action") in {"data_query", "search_sailings"}:
        structured = [
            item
            for item in (state.get("candidates") or [])
            if item.get("source_type") not in {None, "policy"}
        ]
        top_k = max(top_k, min(len(structured) or top_k, 10))
    evidence = rerank_evidence(
        query,
        list(state.get("candidates") or []),
        top_k=top_k,
    )
    return {
        "evidence": evidence,
        "candidates": [],
        "rerank_backend": "lexical_dynamic",
        "status": "evidence_reranked",
        "trace": _trace(
            state,
            "rerank_agent",
            f"Reranked evidence and retained top {len(evidence)}",
            backend="lexical_dynamic",
            top_k=len(evidence),
        ),
    }


def grade_agent(state: ShippingState) -> dict[str, Any]:
    grade, score = grade_evidence(list(state.get("evidence") or []))
    hard_errors = [
        error
        for error in (state.get("errors") or [])
        if not str(error).startswith("Missing route parameters:")
    ]
    if hard_errors:
        grade = "fail"
        score = 0.0
    return {
        "evidence_grade": grade,
        "evidence_score": score,
        "status": f"evidence_{grade}",
        "trace": _trace(
            state,
            "grade_agent",
            f"Evidence grade is {grade}",
            grade=grade,
            score=score,
        ),
    }


def _after_grade(state: ShippingState) -> Literal["rewrite", "generate"]:
    cfg = load_config()
    if (
        state.get("evidence_grade") == "fail"
        and not state.get("errors")
        and int(state.get("retrieval_attempts") or 0)
        <= cfg.max_retrieval_retries
    ):
        return "rewrite"
    return "generate"


def generate_agent(state: ShippingState) -> dict[str, Any]:
    evidence = list(state.get("evidence") or [])
    grade = state.get("evidence_grade") or "fail"
    soft_missing_route = any(
        str(error).startswith("Missing route parameters:")
        for error in (state.get("errors") or [])
    )
    hard_errors = [
        error
        for error in (state.get("errors") or [])
        if not str(error).startswith("Missing route parameters:")
    ]
    if hard_errors:
        answer = _fallback_answer(state)
    elif grade == "fail" or not evidence:
        answer = (
            "I could not find sufficient authoritative shipping evidence to answer "
            "that request. Please provide a booking, quotation, customer, voyage, "
            "route, or policy topic."
        )
    elif soft_missing_route and evidence:
        # Prefer a deterministic sailing-id explanation when the user asked about
        # sailings without a complete route.
        answer = _fallback_answer({**state, "errors": []})
        if evidence and "[S" not in answer:
            answer += " [S1]"
    elif state.get("action") == "data_query" and isinstance(
        state.get("result"), dict
    ) and (state.get("result") or {}).get("entity") in {
        "sailings",
        "quotations",
    }:
        answer = _fallback_answer(state)
        if evidence and "[S" not in answer:
            answer += " [S1]"
    elif not load_config().use_llm_answers:
        answer = _fallback_answer(state)
        if evidence:
            answer += " [S1]"
    else:
        context = evidence_context(evidence)
        try:
            response = get_llm(temperature=0.1, reasoning=False).invoke(
                [
                    SystemMessage(
                        content=(
                            "Answer the shipping request using only the numbered "
                            "evidence. Cite every factual paragraph with [S1], [S2], "
                            "etc. Never invent rates, capacity, dates, references, "
                            "status, or policy. If evidence includes available "
                            "sailings, explain sailing_id values from that data. "
                            "If evidence is weak, state the limitation. Return "
                            "concise plain text."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Request: {state['prompt']}\n"
                            f"Evidence grade: {grade}\n\n{context}"
                        )
                    ),
                ]
            )
            answer = str(response.content).strip()
        except Exception:  # noqa: BLE001
            answer = _fallback_answer(state)
            if evidence:
                answer += " [S1]"
    return {
        "answer": answer,
        "status": "answer_generated",
        "trace": _trace(
            state,
            "generate_agent",
            "Generated an evidence-grounded draft",
            evidence_count=len(evidence),
        ),
    }


def verify_agent(state: ShippingState) -> dict[str, Any]:
    answer = state.get("answer") or ""
    evidence = list(state.get("evidence") or [])
    verified, issues = verify_answer(
        answer,
        evidence,
        grade=state.get("evidence_grade") or "fail",
    )
    return {
        "verified": verified,
        "verification_issues": issues,
        "status": "verified" if verified else "verification_failed",
        "trace": _trace(
            state,
            "verify_agent",
            "Verified citations and shipping references"
            if verified
            else "Answer needs repair before delivery",
            verified=verified,
            issues=issues,
        ),
    }


def _after_verify(state: ShippingState) -> Literal["fix", "response"]:
    if state.get("verified"):
        return "response"
    if int(state.get("fix_attempts") or 0) >= load_config().max_fix_attempts:
        return "response"
    return "fix"


def fix_agent(state: ShippingState) -> dict[str, Any]:
    evidence = list(state.get("evidence") or [])
    context = evidence_context(evidence)
    attempts = int(state.get("fix_attempts") or 0) + 1
    try:
        if not load_config().use_llm_answers:
            raise RuntimeError("LLM answer repair disabled")
        response = get_llm(temperature=0, reasoning=False).invoke(
            [
                SystemMessage(
                    content=(
                        "Repair the answer using only the supplied evidence. Remove "
                        "unsupported references and add valid [S#] citations. Return "
                        "only concise plain text."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Request: {state['prompt']}\n"
                        f"Issues: {state.get('verification_issues')}\n"
                        f"Draft: {state.get('answer')}\n\n{context}"
                    )
                ),
            ]
        )
        answer = str(response.content).strip()
    except Exception:  # noqa: BLE001
        answer = _fallback_answer(state)
        if evidence:
            answer += " [S1]"
    return {
        "answer": answer,
        "fix_attempts": attempts,
        "status": "answer_fixed",
        "trace": _trace(
            state,
            "fix_agent",
            "Repaired unsupported claims and citations",
            fix_attempt=attempts,
        ),
    }


def _after_operations(
    state: ShippingState,
) -> Literal["pricing", "response"]:
    if state.get("errors"):
        return "response"
    return "pricing" if state["action"] in WRITE_ACTIONS else "response"


def pricing_agent(state: ShippingState) -> dict[str, Any]:
    params = state["parameters"]
    try:
        if state["action"] == "create_quotation":
            proposal = repository.build_quotation_proposal(
                params["customer_code"],
                int(params["sailing_id"]),
                params["container_type"],
                int(params["container_qty"]),
                float(params["cargo_weight_kg"]),
                params.get("cargo_description") or "General cargo",
                dangerous_goods=bool(params.get("dangerous_goods")),
            )
        else:
            proposal = repository.build_booking_proposal(params["quote_ref"])
        errors: list[str] = []
    except Exception as exc:  # noqa: BLE001
        proposal = {}
        errors = [str(exc)]
    return {
        "proposal": proposal,
        "errors": errors,
        "status": "invalid_request" if errors else "proposal_ready",
        "trace": _trace(
            state,
            "pricing_agent",
            "Calculated commercial proposal",
            total_usd=proposal.get("total_usd"),
            errors=errors,
        ),
    }


def _after_pricing(
    state: ShippingState,
) -> Literal["risk", "response"]:
    return "response" if state.get("errors") else "risk"


def risk_agent(state: ShippingState) -> dict[str, Any]:
    proposal = state.get("proposal") or {}
    warnings = list(proposal.get("warnings") or [])
    hard_blocks = [
        warning
        for warning in warnings
        if warning.startswith("Capacity shortfall")
        or "Dangerous goods are not allowed" in warning
        or "expired" in warning.lower()
        or "not approved/accepted" in warning
    ]
    review = {
        "risk_level": "high" if hard_blocks else ("medium" if warnings else "low"),
        "warnings": warnings,
        "hard_blocks": hard_blocks,
        "human_approval_required": not hard_blocks,
    }
    errors = list(state.get("errors") or [])
    if hard_blocks:
        errors.append("Compliance hard block: " + "; ".join(hard_blocks))
    return {
        "risk_review": review,
        "errors": errors,
        "status": "blocked" if hard_blocks else "reviewed",
        "trace": _trace(
            state,
            "risk_agent",
            "Reviewed capacity, credit, dangerous goods, and validity",
            risk_review=review,
        ),
    }


def _after_risk(
    state: ShippingState,
) -> Literal["approval_request", "response"]:
    return (
        "response"
        if state.get("risk_review", {}).get("hard_blocks")
        else "approval_request"
    )


def approval_request_agent(state: ShippingState) -> dict[str, Any]:
    approval = repository.create_approval_request(
        state["thread_id"],
        state["action"],
        state["proposal"],
        state["risk_review"],
    )
    return {
        "approval": approval,
        "status": "awaiting_approval",
        "trace": _trace(
            state,
            "approval_agent",
            "Persisted approval request; graph will pause before execution",
            approval_id=approval["approval_id"],
        ),
    }


def execute_agent(state: ShippingState) -> dict[str, Any]:
    try:
        result = repository.execute_approved(state["thread_id"])
        errors = list(state.get("errors") or [])
    except Exception as exc:  # noqa: BLE001
        result = {"executed": False, "status": "execution_error"}
        errors = [*list(state.get("errors") or []), str(exc)]
    return {
        "result": result,
        "errors": errors,
        "status": str(result.get("status") or "execution_complete"),
        "trace": _trace(
            state,
            "execution_agent",
            "Validated approval and attempted the PostgreSQL write",
            result=result,
        ),
    }


def _fallback_answer(state: ShippingState) -> str:
    action = state.get("action")
    errors = state.get("errors") or []
    result = state.get("result")
    proposal = state.get("proposal") or {}

    if errors:
        invalid_errors = [
            error for error in errors if not error.startswith("Missing ")
        ]
        if invalid_errors:
            return (
                "I couldn't validate the supplied shipping details. "
                + " ".join(invalid_errors)
            )
        missing_error = next(
            (error for error in errors if error.startswith("Missing ")),
            None,
        )
        if missing_error:
            missing = missing_error.split(":", 1)[-1].strip()
            request_name = {
                "search_sailings": "search for a sailing",
                "track_booking": "track a booking",
                "create_quotation": "prepare a quotation",
                "create_booking": "create a booking",
            }.get(action, "complete the request")
            return (
                f"I understand that you want me to {request_name}, but I still "
                f"need: {missing}. Send those details in your next message."
            )
        return "I couldn't complete that request. " + " ".join(errors)
    if action == "conversation":
        return (
            "Hello! I’m your shipping operations assistant. I can search sailings, "
            "prepare freight quotations, create bookings after your approval, and "
            "track shipments. How can I help?"
        )
    if action == "data_query":
        data = result if isinstance(result, dict) else {}
        entity = str(data.get("entity") or "shipping records").replace("_", " ")
        if data.get("operation") == "overview":
            return (
                f"The shipping database contains {data.get('total_records', 0)} "
                f"records across {data.get('entity_count', 0)} supported entities. "
                "The structured result includes a status breakdown for each entity."
            )
        if "count" in data:
            breakdown = data.get("by_status")
            detail = f" Status breakdown: {breakdown}." if breakdown else ""
            return (
                f"I found {data.get('count', 0)} matching {entity} records."
                f"{detail}"
            )
        records = data.get("records") if isinstance(data.get("records"), list) else []
        if entity == "quotations" and records:
            latest = records[0]
            total = int(data.get("total_matching", len(records)))
            return (
                f"There {'is' if total == 1 else 'are'} {total} quotation"
                f"{'' if total == 1 else 's'} in the system. The latest is "
                f"{latest.get('quote_ref') or latest.get('reference')} for "
                f"{latest.get('customer_code')} from {latest.get('origin')} to "
                f"{latest.get('destination')}, status {latest.get('status')}, "
                f"total USD {latest.get('total_usd')}, created "
                f"{latest.get('created_at')}."
            )
        if entity == "sailings" and records:
            evidence = state.get("evidence") or []
            evidence_rows = [
                item.get("payload")
                for item in evidence
                if isinstance(item.get("payload"), dict)
                and item.get("payload", {}).get("sailing_id") is not None
            ]
            rows = evidence_rows or records
            lines = []
            for item in rows[:8]:
                lines.append(
                    f"sailing_id {item.get('sailing_id')} · "
                    f"{item.get('voyage_number')} · "
                    f"{item.get('origin') or item.get('origin_code')}"
                    f"→{item.get('destination') or item.get('destination_code')} · "
                    f"{item.get('vessel_name')}"
                )
            return (
                "A sailing_id is the database primary key for a scheduled voyage. "
                f"Here are {len(lines)} sailings from PostgreSQL:\n- "
                + "\n- ".join(lines)
            )
        return (
            f"I retrieved {data.get('returned', 0)} matching {entity} records "
            f"(bounded to {data.get('limit', 0)})."
        )
    if action == "get_quotation":
        data = result if isinstance(result, dict) else {}
        return (
            f"Quotation {data.get('quote_ref')} is {data.get('status')} for "
            f"{data.get('customer_code')}. It covers {data.get('container_qty')} "
            f"× {data.get('container_type')} from {data.get('origin')} to "
            f"{data.get('destination')}, totaling USD {data.get('total_usd')}."
        )
    if action == "search_sailings":
        if isinstance(result, dict):
            matched = result.get("matched") or []
            available = result.get("available_sailings") or []
            if matched:
                first = matched[0]
                return (
                    f"I found {len(matched)} scheduled sailing"
                    f"{'s' if len(matched) != 1 else ''}. The earliest is "
                    f"sailing_id {first.get('sailing_id')} "
                    f"({first['voyage_number']}) on {first['vessel_name']}, "
                    f"departing {first['origin_code']} on {first['departure_at']} "
                    f"and arriving {first['destination_code']} on "
                    f"{first['arrival_at']}. Available capacity is "
                    f"{first['available_teu']} TEU."
                )
            ports = result.get("available_ports") or []
            port_codes = ", ".join(
                str(port.get("unlocode"))
                for port in ports[:8]
                if isinstance(port, dict)
            )
            lines = []
            for item in available[:6]:
                lines.append(
                    f"sailing_id {item.get('sailing_id')} · "
                    f"{item.get('voyage_number')} · "
                    f"{item.get('origin')}→{item.get('destination')}"
                )
            prefix = str(result.get("message") or "No sailing matched that route.")
            if lines:
                return (
                    f"{prefix} Available sample ports: {port_codes or 'n/a'}. "
                    "Available sailings:\n- " + "\n- ".join(lines)
                )
            return prefix
        sailings = result if isinstance(result, list) else []
        if not sailings:
            return "I couldn't find a scheduled sailing for that route."
        first = sailings[0]
        return (
            f"I found {len(sailings)} scheduled sailing"
            f"{'s' if len(sailings) != 1 else ''}. The earliest is "
            f"sailing_id {first.get('sailing_id')} ({first['voyage_number']}) on "
            f"{first['vessel_name']}, departing "
            f"{first['origin_code']} on {first['departure_at']} and arriving "
            f"{first['destination_code']} on {first['arrival_at']}. "
            f"Available capacity is {first['available_teu']} TEU."
        )
    if action == "reference_data":
        data = result if isinstance(result, dict) else {}
        return (
            f"I can work with {len(data.get('customers') or [])} sample customers, "
            f"{len(data.get('ports') or [])} ports, and container types "
            f"{', '.join(data.get('container_types') or [])}."
        )
    if action == "track_booking":
        data = result if isinstance(result, dict) else {}
        return (
            f"Booking {data.get('booking_ref')} is {data.get('status')}. "
            f"It is assigned to voyage {data.get('voyage_number')} from "
            f"{data.get('origin')} to {data.get('destination')}."
        )
    if action == "create_quotation":
        data = result if isinstance(result, dict) else {}
        if data.get("executed"):
            return (
                f"Quotation {data.get('quote_ref')} was created after human "
                f"approval. The approved total is USD {data.get('total_usd')} "
                f"and it is valid until {data.get('valid_until')}."
            )
        if data.get("status") == "rejected":
            return (
                "The quotation proposal was rejected by the human reviewer. "
                "No quotation was written to PostgreSQL."
            )
    if action == "create_booking":
        data = result if isinstance(result, dict) else {}
        if data.get("executed"):
            return (
                f"Booking {data.get('booking_ref')} was confirmed after human "
                f"approval for quotation {data.get('quote_ref')}."
            )
        if data.get("status") == "rejected":
            return (
                "The booking proposal was rejected by the human reviewer. "
                "No booking was created."
            )
    if proposal:
        return (
            "I prepared the requested proposal, but it has not been executed. "
            "Review the structured details and any risk warnings."
        )
    return "The shipping workflow completed. Review the structured result for details."


def _natural_answer(state: ShippingState) -> str:
    fallback = _fallback_answer(state)
    result = state.get("result")
    if (
        state.get("action") in {"create_quotation", "create_booking"}
        and isinstance(result, dict)
        and (
            result.get("executed")
            or result.get("status") in {"executed", "rejected"}
        )
    ):
        # Execution state is authoritative. Do not let the LLM ask for approval
        # again after the human decision has already been applied.
        return fallback
    if (
        state.get("errors")
        or not load_config().use_llm_answers
    ):
        return fallback
    if state.get("action") == "conversation":
        try:
            response = get_llm(temperature=0.3, reasoning=False).invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a friendly shipping chatbot. Respond directly "
                            "to the user's exact conversational message in one or two "
                            "short sentences. If asked how you are, answer how you are. "
                            "Do not introduce yourself or list capabilities unless "
                            "the user explicitly asks what you can do."
                        )
                    ),
                    HumanMessage(content=str(state.get("prompt") or "")),
                ]
            )
            answer = str(response.content).strip()
            return answer or fallback
        except Exception:  # noqa: BLE001
            return fallback
    evidence = {
        "request": state.get("prompt"),
        "action": state.get("action"),
        "status": state.get("status"),
        "result": state.get("result"),
        "proposal": state.get("proposal"),
        "risk_review": state.get("risk_review"),
    }
    try:
        response = get_llm(temperature=0.1, reasoning=False).invoke(
            [
                SystemMessage(
                    content=(
                        "You are a concise shipping operations assistant. "
                        "Answer using only the supplied workflow JSON. Do not invent "
                        "rates, dates, references, capacity, or status. Explain the "
                        "result naturally in 2-5 sentences. If a write was rejected "
                        "or blocked, clearly say no database write occurred. Do not "
                        "mention evidence sections that are absent, empty, or not "
                        "relevant to the user's request. Return plain text without "
                        "Markdown formatting."
                    )
                ),
                HumanMessage(
                    content=json.dumps(evidence, default=str, ensure_ascii=False)
                ),
            ]
        )
        answer = str(response.content).strip()
        return answer or fallback
    except Exception:  # noqa: BLE001
        return fallback


def response_agent(state: ShippingState) -> dict[str, Any]:
    answer = state.get("answer") or _natural_answer(state)
    if (
        state.get("lane") == "rag"
        and not state.get("verified")
        and not answer
    ):
        answer = (
            "I retrieved evidence but could not verify a fully supported answer. "
            "Please narrow the request or provide an exact shipping reference."
        )
    elif state.get("lane") == "rag" and not state.get("verified"):
        # Prefer the grounded draft/fallback over a generic refusal when we still
        # have structured evidence (for example sailing catalogs).
        if not answer or "could not verify a fully supported answer" in answer:
            answer = _fallback_answer({**state, "errors": []}) or answer
            evidence = list(state.get("evidence") or [])
            if evidence and answer and "[S" not in answer:
                answer += " [S1]"
    response = {
        "thread_id": state["thread_id"],
        "action": state.get("action"),
        "status": state.get("status"),
        "answer": answer,
        "data": state.get("result") or state.get("proposal") or {},
        "errors": state.get("errors") or [],
        "approval": state.get("approval") or None,
        "verified": state.get("verified") if state.get("lane") == "rag" else None,
        "evidence_grade": state.get("evidence_grade") or "n/a",
        "citations": [
            {
                "id": item.get("citation"),
                "source_type": item.get("source_type"),
                "source_id": item.get("source_id"),
                "title": item.get("title"),
                "score": item.get("rerank_score"),
            }
            for item in state.get("evidence") or []
        ],
    }
    return {
        "response": response,
        "answer": answer,
        "trace": _trace(
            state,
            "response_agent",
            "Built structured JSON response",
            status=response["status"],
        ),
    }


def build_graph(checkpointer: SqliteSaver):
    builder = StateGraph(ShippingState)
    builder.add_node("understand", supervisor_agent)
    builder.add_node("rewrite", rewrite_agent)
    builder.add_node("retrieve", retrieve_agent)
    builder.add_node("rerank", rerank_agent)
    builder.add_node("grade", grade_agent)
    builder.add_node("generate", generate_agent)
    builder.add_node("verify", verify_agent)
    builder.add_node("fix", fix_agent)
    builder.add_node("operations", operations_agent)
    builder.add_node("pricing", pricing_agent)
    builder.add_node("risk", risk_agent)
    builder.add_node("approval_request", approval_request_agent)
    builder.add_node("execute", execute_agent)
    builder.add_node("response", response_agent)

    builder.add_edge(START, "understand")
    builder.add_conditional_edges(
        "understand",
        _after_understand,
        {
            "response": "response",
            "rewrite": "rewrite",
            "operations": "operations",
        },
    )
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "grade")
    builder.add_conditional_edges(
        "grade",
        _after_grade,
        {"rewrite": "rewrite", "generate": "generate"},
    )
    builder.add_edge("generate", "verify")
    builder.add_conditional_edges(
        "verify",
        _after_verify,
        {"fix": "fix", "response": "response"},
    )
    builder.add_edge("fix", "verify")
    builder.add_conditional_edges(
        "operations",
        _after_operations,
        {"pricing": "pricing", "response": "response"},
    )
    builder.add_conditional_edges(
        "pricing",
        _after_pricing,
        {"risk": "risk", "response": "response"},
    )
    builder.add_conditional_edges(
        "risk",
        _after_risk,
        {"approval_request": "approval_request", "response": "response"},
    )
    builder.add_edge("approval_request", "execute")
    builder.add_edge("execute", "response")
    builder.add_edge("response", END)
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["execute"],
    )


_CHECKPOINT_PATH = load_config().checkpoint_db
_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
_CHECKPOINT_CONNECTION = sqlite3.connect(
    _CHECKPOINT_PATH,
    check_same_thread=False,
)
_GRAPH = build_graph(SqliteSaver(_CHECKPOINT_CONNECTION))


def _config(thread_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 40,
    }


def _initial_state(prompt: str, thread_id: str) -> ShippingState:
    return {
        "thread_id": thread_id,
        "prompt": prompt,
        "action": "",
        "plan": [],
        "parameters": {},
        "proposal": {},
        "risk_review": {},
        "approval": {},
        "result": {},
        "response": {},
        "status": "new",
        "errors": [],
        "trace": [],
        "lane": "",
        "route_reason": "",
        "rewritten_query": "",
        "search_queries": [],
        "candidates": [],
        "evidence": [],
        "evidence_grade": "n/a",
        "evidence_score": 0.0,
        "rerank_backend": "",
        "retrieval_attempts": 0,
        "answer": "",
        "verified": False,
        "verification_issues": [],
        "fix_attempts": 0,
    }


def _pending_answer(state: dict[str, Any]) -> str:
    proposal = state.get("proposal") or {}
    action = state.get("action")
    warnings = (state.get("risk_review") or {}).get("warnings") or []
    if action == "create_quotation":
        message = (
            f"I prepared a quotation proposal for {proposal.get('customer_code')} "
            f"on voyage {proposal.get('voyage_number')}, from "
            f"{proposal.get('origin')} to {proposal.get('destination')}. "
            f"The proposed total is {proposal.get('currency', 'USD')} "
            f"{proposal.get('total_usd')}."
        )
    else:
        message = (
            f"I prepared a booking proposal for quotation "
            f"{proposal.get('quote_ref')} on voyage {proposal.get('voyage_number')}."
        )
    if warnings:
        message += " Review warning: " + "; ".join(warnings) + "."
    return message + " Human approval is required before I write this to PostgreSQL."


def _recovery_choices(state: dict[str, Any]) -> list[dict[str, str]]:
    if state.get("status") != "invalid_request":
        return []
    action = state.get("action")
    params = state.get("parameters") or {}
    errors = [str(error) for error in state.get("errors") or []]
    error_text = " ".join(errors).lower()
    choices: list[dict[str, str]] = []

    references = repository.list_reference_data()
    known_customers = {
        str(customer.get("customer_code"))
        for customer in references.get("customers") or []
    }
    needs_customer = action == "create_quotation" and (
        not params.get("customer_code")
        or params.get("customer_code") not in known_customers
        or "customer" in error_text
    )
    if needs_customer:
        for customer in references.get("customers") or []:
            code = str(customer.get("customer_code"))
            choices.append(
                {
                    "kind": "customer",
                    "label": f"{code} — {customer.get('name')}",
                    "value": f"customer_code: {code}",
                }
            )

    needs_sailing = action in {"create_quotation", "search_sailings"} and (
        not params.get("sailing_id")
        or "sailing" in error_text
        or "voyage" in error_text
        or "route" in error_text
    )
    if needs_sailing:
        sailings = (
            repository.query_shipping_data(
                "sailings",
                operation="list",
                status="scheduled",
                limit=25,
            ).get("records")
            or []
        )
        origin = params.get("origin")
        destination = params.get("destination")
        route_matches = [
            sailing
            for sailing in sailings
            if (not origin or sailing.get("origin") == origin)
            and (
                not destination
                or sailing.get("destination") == destination
            )
        ]
        selected_sailings = route_matches or sailings
        for sailing in selected_sailings[:8]:
            voyage = str(sailing.get("voyage_number"))
            sailing_origin = str(sailing.get("origin"))
            sailing_destination = str(sailing.get("destination"))
            choices.append(
                {
                    "kind": "sailing",
                    "label": (
                        f"{voyage} — {sailing_origin} → "
                        f"{sailing_destination}"
                    ),
                    "value": (
                        f"voyage_number: {voyage}, "
                        f"origin: {sailing_origin}, "
                        f"destination: {sailing_destination}"
                    ),
                }
            )
    return choices


def _public_result(
    state: dict[str, Any],
    thread_id: str,
    *,
    interrupted: bool,
) -> dict[str, Any]:
    response = state.get("response") or None
    assistant_message = (
        _pending_answer(state)
        if interrupted
        else (
            response.get("answer")
            if isinstance(response, dict)
            else "The workflow completed."
        )
    )
    choices = _recovery_choices(state)
    if choices:
        lines = "\n".join(
            f"- {choice['label']}" for choice in choices
        )
        assistant_message = (
            f"{assistant_message}\n\nChoose a valid option below:\n{lines}"
        )
    return {
        "thread_id": thread_id,
        "interrupted": interrupted,
        "pending": state.get("approval") if interrupted else None,
        "assistant_message": assistant_message,
        "choices": choices,
        "response": response,
        "state": {
            "action": state.get("action"),
            "plan": state.get("plan") or [],
            "lane": state.get("lane"),
            "route_reason": state.get("route_reason"),
            "rewritten_query": state.get("rewritten_query"),
            "evidence_grade": state.get("evidence_grade") or "n/a",
            "evidence_score": state.get("evidence_score") or 0.0,
            "rerank_backend": state.get("rerank_backend") or "",
            "retrieval_attempts": state.get("retrieval_attempts") or 0,
            "verified": bool(state.get("verified")),
            "verification_issues": state.get("verification_issues") or [],
            "fix_attempts": state.get("fix_attempts") or 0,
            "parameters": state.get("parameters") or {},
            "proposal": state.get("proposal") or {},
            "risk_review": state.get("risk_review") or {},
            "status": state.get("status"),
            "errors": state.get("errors") or [],
        },
        "evidence": [
            {
                "citation": item.get("citation"),
                "source_type": item.get("source_type"),
                "source_id": item.get("source_id"),
                "title": item.get("title"),
                "score": item.get("rerank_score"),
                "content": item.get("content"),
            }
            for item in state.get("evidence") or []
        ],
        "trace": state.get("trace") or [],
        "graph": graph_topology(),
    }


def run_prompt(prompt: str, *, thread_id: str | None = None) -> dict[str, Any]:
    text = (prompt or "").strip()
    if not text:
        raise ValueError("prompt is required")
    resolved_thread = thread_id or str(uuid4())
    config = _config(resolved_thread)
    state = _GRAPH.invoke(_initial_state(text, resolved_thread), config)
    snapshot = _GRAPH.get_state(config)
    interrupted = "execute" in snapshot.next
    return _public_result(state, resolved_thread, interrupted=interrupted)


def resume(
    thread_id: str,
    *,
    approve: bool,
    reviewer: str,
    note: str = "",
) -> dict[str, Any]:
    config = _config(thread_id)
    snapshot = _GRAPH.get_state(config)
    if "execute" not in snapshot.next:
        raise ValueError("Thread is not waiting for approval")
    decision = repository.decide_approval(
        thread_id,
        approve=approve,
        reviewer=reviewer,
        note=note,
    )
    _GRAPH.update_state(
        config,
        {
            "approval": decision,
            "status": "approved" if approve else "rejected",
        },
    )
    state = _GRAPH.invoke(None, config)
    return _public_result(state, thread_id, interrupted=False)


def graph_topology() -> dict[str, Any]:
    return {
        "nodes": GRAPH_NODES,
        "edges": GRAPH_EDGES,
        "mermaid": MERMAID,
        "agents": [
            "understand",
            "rewrite",
            "retrieve",
            "rerank",
            "grade",
            "generate",
            "verify",
            "fix",
            "operations",
            "pricing",
            "risk",
            "response",
        ],
        "lanes": ["chat", "rag", "write"],
        "max_retrieval_retries": load_config().max_retrieval_retries,
        "max_fix_attempts": load_config().max_fix_attempts,
        "mcp_tools": [
            "shipping_reference_data",
            "shipping_data_overview",
            "shipping_policy_knowledge",
            "query_shipping_data",
            "search_sailings",
            "calculate_quotation",
            "get_quotation",
            "track_booking",
            "execute_human_approved_action",
        ],
        "human_approval_before": ["create_quotation", "create_booking"],
    }

