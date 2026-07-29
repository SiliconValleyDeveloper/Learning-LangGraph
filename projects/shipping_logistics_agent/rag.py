"""Structured-evidence RAG helpers for the shipping LangGraph.

PostgreSQL remains authoritative for operational facts. Static policy evidence
documents explain business rules that are implemented by the repository layer.
"""

from __future__ import annotations

import json
import re
from typing import Any

from projects.advanced_chatbot.rerank import lexical_rerank_score

POLICY_EVIDENCE = [
    {
        "source_id": "policy:sailing-id",
        "title": "Sailing ID definition",
        "content": (
            "A sailing_id is the PostgreSQL primary key for shipping.sailings. "
            "Each sailing also has a voyage_number, vessel, origin UN/LOCODE, "
            "destination UN/LOCODE, departure time, arrival time, available TEU, "
            "and base freight rates. Use sailing_id when creating quotations."
        ),
    },
    {
        "source_id": "policy:human-approval",
        "title": "Human approval policy",
        "content": (
            "Quotation and booking writes require an explicit human approval. "
            "LangGraph pauses before execution, the decision is persisted, and "
            "the repository validates approval again before PostgreSQL is mutated."
        ),
    },
    {
        "source_id": "policy:capacity",
        "title": "Container capacity policy",
        "content": (
            "A 20GP container consumes 1 TEU. 40GP, 40HC, and 40RF containers "
            "consume 2 TEU each. A proposal is blocked when required TEU exceeds "
            "the sailing's available TEU."
        ),
    },
    {
        "source_id": "policy:dangerous-goods",
        "title": "Dangerous goods policy",
        "content": (
            "Dangerous goods must be declared. A quotation is hard-blocked when "
            "dangerous goods are requested on a sailing that does not allow them."
        ),
    },
    {
        "source_id": "policy:quotation-pricing",
        "title": "Quotation pricing policy",
        "content": (
            "Ocean freight uses the sailing's 20-foot or 40-foot base rate times "
            "container quantity. Standard surcharges are 12 percent of ocean "
            "freight. 40RF also adds the sailing's reefer surcharge per container."
        ),
    },
    {
        "source_id": "policy:booking",
        "title": "Booking eligibility policy",
        "content": (
            "A booking can execute only from an approved or accepted quotation "
            "that has not expired and when the sailing still has enough capacity."
        ),
    },
]


def _source_id(payload: dict[str, Any], action: str, index: int) -> str:
    for key in (
        "booking_ref",
        "quote_ref",
        "voyage_number",
        "customer_code",
        "unlocode",
        "container_number",
        "reference",
        "event_code",
    ):
        if payload.get(key):
            return str(payload[key])
    return f"{action}:{index}"


def _candidate(
    action: str,
    payload: dict[str, Any],
    index: int,
    *,
    source_type: str | None = None,
) -> dict[str, Any]:
    source_id = _source_id(payload, action, index)
    title = f"{action.replace('_', ' ').title()} — {source_id}"
    return {
        "source": source_id,
        "source_type": source_type or action,
        "source_id": source_id,
        "title": title,
        "content": f"{title}\n{json.dumps(payload, default=str, ensure_ascii=False)}",
        "score": 0.75,
        "payload": payload,
    }


def build_candidates(
    action: str,
    result: dict[str, Any] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert safe repository output into uniform evidence candidates."""
    candidates: list[dict[str, Any]] = []
    if isinstance(result, list):
        candidates.extend(
            _candidate(action, item, index)
            for index, item in enumerate(result, start=1)
        )
    elif isinstance(result, dict):
        records = result.get("records")
        available = result.get("available_sailings")
        matched = result.get("matched")
        if isinstance(matched, list) and matched:
            candidates.extend(
                _candidate(action, item, index)
                for index, item in enumerate(matched, start=1)
            )
        elif isinstance(records, list):
            candidates.append(
                _candidate(
                    action,
                    {
                        "entity": result.get("entity"),
                        "operation": result.get("operation"),
                        "filters": result.get("filters") or {},
                        "total_matching": result.get(
                            "total_matching", len(records)
                        ),
                        "returned": result.get("returned", len(records)),
                    },
                    0,
                    source_type=f"{result.get('entity') or action}_summary",
                )
            )
            candidates.extend(
                _candidate(
                    action,
                    item,
                    index,
                    source_type=str(result.get("entity") or action),
                )
                for index, item in enumerate(records, start=1)
                if isinstance(item, dict)
            )
        elif isinstance(available, list) and available:
            explanation = {
                "message": result.get("message"),
                "origin": result.get("origin"),
                "destination": result.get("destination"),
                "available_ports": [
                    port.get("unlocode")
                    for port in (result.get("available_ports") or [])
                    if isinstance(port, dict)
                ],
                "sailing_id_definition": (
                    "sailing_id is the PostgreSQL primary key for shipping.sailings"
                ),
            }
            candidates.append(
                _candidate(action, explanation, 1, source_type="sailing_catalog")
            )
            candidates.extend(
                _candidate(action, item, index + 1, source_type="sailings")
                for index, item in enumerate(available, start=1)
                if isinstance(item, dict)
            )
        elif action == "reference_data":
            index = 0
            for entity in ("customers", "ports"):
                for item in result.get(entity) or []:
                    if isinstance(item, dict):
                        index += 1
                        candidates.append(
                            _candidate(action, item, index, source_type=entity)
                        )
            candidates.append(
                _candidate(
                    action,
                    {"container_types": result.get("container_types") or []},
                    index + 1,
                    source_type="container_types",
                )
            )
        elif action == "track_booking":
            base = {key: value for key, value in result.items() if key != "events"}
            candidates.append(_candidate(action, base, 1, source_type="bookings"))
            for index, event in enumerate(result.get("events") or [], start=2):
                candidates.append(
                    _candidate(action, event, index, source_type="shipment_events")
                )
        elif result:
            candidates.append(_candidate(action, result, 1))
    return candidates


def policy_candidates() -> list[dict[str, Any]]:
    return [
        {
            "source": item["source_id"],
            "source_type": "policy",
            "source_id": item["source_id"],
            "title": item["title"],
            "content": item["content"],
            "score": 0.0,
            "payload": {},
        }
        for item in POLICY_EVIDENCE
    ]


def rerank_evidence(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        item["rerank_score"] = lexical_rerank_score(
            query,
            str(item.get("content") or ""),
            float(item.get("score") or 0.0),
        )
        scored.append(item)
    scored.sort(
        key=lambda item: float(item.get("rerank_score") or 0.0),
        reverse=True,
    )
    ranked = scored[: max(1, top_k)]
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
        item["citation"] = f"S{index}"
    return ranked


def grade_evidence(evidence: list[dict[str, Any]]) -> tuple[str, float]:
    if not evidence:
        return "fail", 0.0
    best = max(float(item.get("rerank_score") or 0.0) for item in evidence)
    has_structured = any(item.get("source_type") != "policy" for item in evidence)
    if has_structured or best >= 0.12:
        return "pass", round(best, 4)
    if best >= 0.05:
        return "weak", round(best, 4)
    return "fail", round(best, 4)


def evidence_context(evidence: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[S{index}] {item.get('title')}\n{item.get('content')}"
        for index, item in enumerate(evidence, start=1)
    )


def verify_answer(
    answer: str,
    evidence: list[dict[str, Any]],
    *,
    grade: str,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    citations = [int(value) for value in re.findall(r"\[S(\d+)\]", answer)]
    if evidence and grade != "fail" and not citations:
        issues.append("Answer has no evidence citation")
    invalid = [number for number in citations if number < 1 or number > len(evidence)]
    if invalid:
        issues.append(f"Invalid citations: {invalid}")

    context = evidence_context(evidence).upper()
    upper_answer = answer.upper()
    known_codes = {"INNSA", "SGSIN", "AEDXB", "NLRTM", "USNYC", "USMIA", "GBLGP"}
    for item in evidence:
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        for key in (
            "voyage_number",
            "origin",
            "destination",
            "origin_code",
            "destination_code",
            "unlocode",
            "quote_ref",
            "booking_ref",
            "reference",
        ):
            value = payload.get(key)
            if value:
                known_codes.add(str(value).upper())
        content = str(item.get("content") or "").upper()
        known_codes.update(re.findall(r"\b(?:SL[QB]-[A-Z0-9]+|[A-Z]{2}\d{3}[A-Z])\b", content))

    references = set(re.findall(r"\bSL[QB]-[A-Z0-9]+\b", upper_answer))
    references.update(code for code in known_codes if code in upper_answer and len(code) == 5)
    # Voyage numbers mentioned in the answer must appear in evidence.
    references.update(re.findall(r"\b[A-Z]{2}\d{3}[A-Z]\b", upper_answer))

    unsupported = sorted(
        reference
        for reference in references
        if reference not in context and reference not in known_codes
    )
    if unsupported:
        issues.append(f"Unsupported references: {unsupported}")
    return not issues, issues
