"""Prompt-to-parameters extraction with deterministic fallback."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from Learning.llm import get_llm
from projects.shipping_logistics_agent import repository
from projects.shipping_logistics_agent.resolvers import (
    PORT_ALIASES as _PORT_ALIASES,
    apply_patches,
    parse_choice_value,
    resolve_customer,
    resolve_port,
)
_PARAM_KEYS = {
    "customer_code",
    "origin",
    "destination",
    "departure_after",
    "sailing_id",
    "voyage_number",
    "container_type",
    "container_qty",
    "cargo_weight_kg",
    "cargo_description",
    "dangerous_goods",
    "quote_ref",
    "booking_ref",
    "entity",
    "query_mode",
    "status",
    "limit",
    "reference",
    "include_amounts",
    "unrecognized_origin",
    "unrecognized_destination",
}
_ACTIONS = {
    "conversation",
    "reference_data",
    "data_query",
    "get_quotation",
    "search_sailings",
    "track_booking",
    "create_quotation",
    "create_booking",
}
_ENTITY_TERMS = {
    "customers": ("customer", "customers"),
    "ports": ("port", "ports"),
    "vessels": ("vessel", "vessels", "ships"),
    "sailings": ("sailing", "sailings", "voyages"),
    "quotations": ("quote", "quotes", "quotation", "quotations"),
    "bookings": ("booking", "bookings"),
    "containers": ("container", "containers"),
    "shipment_events": ("event", "events", "shipment events"),
}
_KNOWN_STATUSES = set().union(*repository.ENTITY_STATUSES.values())


def _llm_classify_action(prompt: str) -> str:
    """Legacy single-action classifier kept as a thin wrapper."""
    intent = _llm_resolve_intent(prompt, history=[])
    action = intent.get("action")
    return action if action in _ACTIONS else "reference_data"


def _format_history(history: list[dict[str, str]] | None) -> str:
    if not history:
        return "(no prior turns)"
    lines: list[str] = []
    for item in history[-6:]:
        role = str(item.get("role") or "user").strip().lower()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content[:500]}")
    return "\n".join(lines) if lines else "(no prior turns)"


def _llm_resolve_intent(
    prompt: str,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Ask the local LLM to resolve action + lane hints from chat context."""
    system = (
        "You are the intent router for a shipping logistics agent. "
        "Return ONLY valid JSON with keys: action, lane, entity, query_mode, "
        "status, limit, follow_up, reason. "
        "action must be one of: conversation, reference_data, data_query, "
        "get_quotation, search_sailings, track_booking, create_quotation, "
        "create_booking. "
        "lane must be one of: chat, rag, db, write. "
        "Use db for counts, lists, amounts, totals, and exact ref lookups. "
        "Use rag for explanations, definitions, and soft sailing discovery. "
        "Use write only for explicit create/prepare/make booking or quotation. "
        "Use chat for greetings. "
        "Read follow-ups such as 'more detail', 'those bookings', 'and the amounts' "
        "must stay data_query on db, never create_booking/create_quotation. "
        "entity must be customers, ports, vessels, sailings, quotations, bookings, "
        "containers, shipment_events, all, or null. "
        "query_mode must be count, list, summary, or null. "
        "status/limit may be null. follow_up is boolean. reason is a short string. "
        "Never invent quote_ref or booking_ref values."
    )
    human = (
        f"Recent conversation:\n{_format_history(history)}\n\n"
        f"Current user message:\n{prompt}"
    )
    response = get_llm(temperature=0, reasoning=False).invoke(
        [SystemMessage(content=system), HumanMessage(content=human)]
    )
    return _json_object(str(response.content))


def _has_follow_up_intent(text: str) -> bool:
    return bool(
        re.search(
            r"\b(more detail|more details|tell me more|elaborate|that one|those|"
            r"the same|same ones?|and the|also show|what about|continue|"
            r"matching\b.{0,40}\brecords?)\b",
            text,
            re.IGNORECASE,
        )
    )


def _needs_llm_router(
    prompt: str,
    history: list[dict[str, str]] | None,
    rules_action: str | None,
) -> bool:
    """Use the LLM when rules are unsure or the turn depends on chat context."""
    text = (prompt or "").lower().strip()
    if rules_action is None:
        return True
    if history and _has_follow_up_intent(text):
        return True
    if history and len(text) < 48 and rules_action in {
        "conversation",
        "reference_data",
        "create_booking",
        "create_quotation",
    }:
        return True
    if history and rules_action in {"create_booking", "create_quotation"} and (
        _has_read_intent(text) or _has_db_fact_intent(text)
    ):
        return True
    return False


def _apply_intent_rails(
    *,
    prompt: str,
    rules_action: str | None,
    llm_intent: dict[str, Any],
) -> dict[str, Any]:
    """Merge rules + LLM, with hard rails for reads vs writes and lanes."""
    text = (prompt or "").lower()
    llm_action = llm_intent.get("action")
    action = (
        llm_action
        if llm_action in _ACTIONS
        else (rules_action or "reference_data")
    )

    # Never let a read/detail follow-up become a write.
    if action in {"create_booking", "create_quotation"} and (
        _has_read_intent(text) or _has_db_fact_intent(text) or _has_follow_up_intent(text)
    ):
        if not (
            _has_create_quotation_intent(text)
            or _has_create_booking_intent(text)
        ):
            action = "data_query"

    # Prefer explicit create rules when present.
    if rules_action == "create_quotation" and _has_create_quotation_intent(text):
        action = "create_quotation"
    if rules_action == "create_booking" and _has_create_booking_intent(text):
        action = "create_booking"
    if rules_action in {
        "get_quotation",
        "track_booking",
        "search_sailings",
        "data_query",
        "conversation",
    } and not llm_intent.get("follow_up"):
        # Keep strong deterministic reads unless this is a contextual follow-up.
        if rules_action and not _has_follow_up_intent(text):
            action = rules_action

    hints: dict[str, Any] = {}
    for key in ("entity", "query_mode", "status", "limit"):
        value = llm_intent.get(key)
        if value in (None, "", [], {}):
            continue
        hints[key] = value
    if hints.get("entity") == "all" or hints.get("entity") in _ENTITY_TERMS:
        pass
    elif "entity" in hints:
        hints.pop("entity", None)
    if hints.get("query_mode") not in {None, "count", "list", "summary"}:
        hints.pop("query_mode", None)
    if hints.get("status") and str(hints["status"]).lower() not in _KNOWN_STATUSES:
        hints.pop("status", None)
    if "limit" in hints:
        try:
            hints["limit"] = min(max(int(hints["limit"]), 1), 25)
        except (TypeError, ValueError):
            hints.pop("limit", None)

    if action == "data_query" and _has_follow_up_intent(text):
        hints.setdefault("query_mode", "list")

    lane = resolve_lane(action, prompt, hints)
    llm_lane = llm_intent.get("lane")
    if llm_lane in {"chat", "rag", "db", "write"}:
        # Soft preference from LLM, but never override write safety rails.
        if action in {"create_quotation", "create_booking"}:
            lane = "write"
        elif action == "conversation":
            lane = "chat"
        elif llm_lane == "db" and action in {
            "data_query",
            "get_quotation",
            "track_booking",
            "search_sailings",
        }:
            lane = "db"
        elif llm_lane == "rag" and action in {
            "data_query",
            "reference_data",
            "search_sailings",
        }:
            # Keep explanations on RAG; keep amount/list on DB.
            if not (
                _has_db_fact_intent(text)
                or hints.get("include_amounts")
                or hints.get("query_mode") in {"count", "list", "summary"}
            ):
                lane = "rag"

    source = "hybrid" if llm_intent else "rules"
    if not rules_action and llm_intent:
        source = "llm"
    if rules_action and not llm_intent:
        source = "rules"

    return {
        "action": action,
        "lane": lane,
        "param_hints": hints,
        "follow_up": bool(llm_intent.get("follow_up") or _has_follow_up_intent(text)),
        "reason": str(llm_intent.get("reason") or f"Routed by {source}"),
        "source": source,
    }


def resolve_intent(
    prompt: str,
    *,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Hybrid intent router: deterministic rules first, LLM for ambiguous turns."""
    rules_action = classify_action_rules(prompt)
    llm_intent: dict[str, Any] = {}
    use_llm = _needs_llm_router(prompt, history, rules_action)
    if use_llm:
        try:
            from projects.shipping_logistics_agent.config import load_config

            if load_config().use_llm_answers:
                llm_intent = _llm_resolve_intent(prompt, history)
        except Exception:  # noqa: BLE001
            llm_intent = {}
    return _apply_intent_rails(
        prompt=prompt,
        rules_action=rules_action,
        llm_intent=llm_intent,
    )


def classify_action_rules(prompt: str) -> str | None:
    """Deterministic action classification, or None when ambiguous."""
    text = prompt.lower()
    if re.fullmatch(
        r"\s*(hi|hello|hey|good\s+(morning|afternoon|evening)|thanks|thank you|bye)[!.?\s]*",
        text,
    ):
        return "conversation"
    if any(phrase in text for phrase in ("data overview", "database overview", "all data")):
        return "data_query"
    quote_ref = re.search(r"\bSLQ-[A-Z0-9]+\b", prompt, re.IGNORECASE)
    if quote_ref and not any(word in text for word in ("book", "booking", "confirm")):
        return "get_quotation"
    if re.search(r"\bSLB-[A-Z0-9]+\b", prompt, re.IGNORECASE) or any(
        phrase in text for phrase in ("track booking", "where is", "shipment status")
    ):
        return "track_booking"
    if re.search(r"\bSLQ-[A-Z0-9]+\b", prompt, re.IGNORECASE) and any(
        word in text for word in ("book", "booking", "confirm")
    ):
        return "create_booking"
    data_terms = any(
        re.search(rf"\b{re.escape(term)}\b", text)
        for terms in _ENTITY_TERMS.values()
        for term in terms
    )
    quotation_terms = any(
        re.search(rf"\b{re.escape(term)}\b", text)
        for term in _ENTITY_TERMS["quotations"]
    )
    booking_terms = any(
        re.search(rf"\b{re.escape(term)}\b", text)
        for term in _ENTITY_TERMS["bookings"]
    )
    if quotation_terms and _has_read_intent(text):
        return "data_query"
    if booking_terms and _has_read_intent(text):
        return "data_query"
    if _has_create_quotation_intent(text):
        return "create_quotation"
    if _has_route_intent(prompt) or any(
        phrase in text
        for phrase in (
            "find sailing",
            "search sailing",
            "sailings from",
            "sailing from",
            "schedule from",
            "voyage from",
        )
    ):
        return "search_sailings"
    if data_terms and _has_read_intent(text):
        return "data_query"
    if data_terms and any(
        phrase in text
        for phrase in (
            "how many",
            "count",
            "summary",
            "overview",
            "list",
            "show all",
            "show me",
            "recent",
            "latest",
            "what is",
            "what's",
            "whats",
            "explain",
            "define",
            "meaning",
            "detail",
            "details",
            "more detail",
            "more details",
        )
    ):
        return "data_query"
    if re.search(
        r"\b(sailing\s*ids?|what(?:'s| is| are)?\s+sailing|show\s+sailings?|"
        r"available\s+sailings?|all\s+sailings?)\b",
        text,
    ):
        return "data_query"
    if any(
        word in text
        for word in ("sailing", "schedule", "voyage", "vessel", "depart")
    ):
        return "data_query"
    if _has_create_booking_intent(text):
        return "create_booking"
    if booking_terms:
        return "data_query"
    if any(word in text for word in ("port", "customer", "container type", "help")):
        return "reference_data"
    return None


def classify_action(
    prompt: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    return str(resolve_intent(prompt, history=history)["action"])


def _has_route_intent(text: str) -> bool:
    if re.search(
        r"\b(?:from|between)\s+[A-Za-z].{0,40}\b(?:to|->|→)\b|"
        r"\b[A-Z]{3,5}\s*(?:to|->|→|-)\s*[A-Z]{3,5}\b|"
        r"\b(?:mia|miami|lon|london|innsa|singapore|dubai|rotterdam|nyc|"
        r"new york|usmia|gblgp|sgsin|aedxb|nlrtm|usnyc)\b"
        r".{0,20}\b(?:to|->|→)\b.{0,20}"
        r"\b(?:mia|miami|lon|london|innsa|singapore|dubai|rotterdam|nyc|"
        r"new york|usmia|gblgp|sgsin|aedxb|nlrtm|usnyc|[a-z]{3,5})\b|"
        r"additional information from the user:.*\b(?:to|->|→)\b",
        text,
        re.IGNORECASE | re.DOTALL,
    ):
        return True
    refs = repository.list_reference_data()
    return len(_port_from_text(text, refs)) >= 2


def _has_read_intent(text: str) -> bool:
    return bool(
        re.search(
            r"\b(how many|count|summary|overview|list|show|display|fetch|find|"
            r"recent|latest|last|newest|oldest|history|all|existing|current|"
            r"detail|details|more detail|more details|tell me more|elaborate)\b|"
            r"\bmatching\b.{0,40}\brecords?\b",
            text,
            re.IGNORECASE,
        )
    )


def _has_create_booking_intent(text: str) -> bool:
    """True only for explicit booking writes, not listing existing bookings."""
    if re.search(
        r"\b(how many|count|list|show|display|detail|details|track|status|"
        r"matching|existing|confirmed bookings?|recent bookings?)\b",
        text,
        re.IGNORECASE,
    ) and not re.search(
        r"\b(create|prepare|make|book this|book the|confirm booking|"
        r"convert|raise a booking)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    return bool(
        re.search(
            r"\b(create|prepare|make|raise|confirm)\b.{0,40}\bbooking\b|"
            r"\bbook\b.{0,20}\b(this|the|quotation|quote|for)\b|"
            r"\b(book it|book now|convert (?:the )?quote|convert (?:the )?quotation)\b",
            text,
            re.IGNORECASE | re.DOTALL,
        )
    )


def _has_create_quotation_intent(text: str) -> bool:
    return bool(
        re.search(
            r"\b(create|prepare|generate|make|request|raise|need|want)\b"
            r".{0,40}\b(quote|quotation)\b|"
            r"\b(quote|quotation)\b.{0,20}\b(for|from)\b|"
            r"\b(quote me|freight cost|rate for|price for)\b",
            text,
            re.IGNORECASE | re.DOTALL,
        )
    )


def _has_db_fact_intent(text: str) -> bool:
    """True when the user wants authoritative SQL facts/aggregates, not explanation."""
    return bool(
        re.search(
            r"\b(how many|count|list|show|display|fetch|latest|last|recent|"
            r"newest|oldest|top\s+\d+|amount|amounts|total|totals|sum|"
            r"calculate|calc|value|values|price|prices)\b",
            text,
            re.IGNORECASE,
        )
    )


def _has_explain_intent(text: str) -> bool:
    return bool(
        re.search(
            r"\b(what is|what's|whats|explain|define|meaning|difference|"
            r"how does|why|policy|help me understand)\b",
            text,
            re.IGNORECASE,
        )
    )


def resolve_lane(
    action: str,
    prompt: str,
    params: dict[str, Any] | None = None,
) -> str:
    """Choose chat | rag | db | write based on action and user intent.

    `db` is for authoritative PostgreSQL list/count/amount/ref lookups.
    `rag` is for explanation, policy, and soft discovery that needs evidence narrative.
    """
    params = params or {}
    if action == "conversation":
        return "chat"
    if action in {"create_quotation", "create_booking"}:
        return "write"

    text = (prompt or "").lower()
    wants_db = (
        _has_db_fact_intent(text)
        or bool(params.get("include_amounts"))
        or str(params.get("query_mode") or "") in {"count", "list", "summary"}
    )
    wants_explain = _has_explain_intent(text)

    if action in {"get_quotation", "track_booking"}:
        if params.get("quote_ref") or params.get("booking_ref") or wants_db:
            return "db"
        return "rag"

    if action == "data_query":
        if wants_explain and re.search(
            r"\b(what is|what's|whats|explain|define|meaning)\b", text
        ):
            return "rag"
        if wants_db or params.get("entity"):
            return "db"
        return "rag"

    if action == "search_sailings":
        if wants_db and not wants_explain:
            return "db"
        return "rag"

    if action == "reference_data":
        return "rag"
    return "rag"


def _json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def _llm_extract(prompt: str, action: str) -> dict[str, Any]:
    refs = repository.list_reference_data()
    system = (
        "Extract shipping request parameters. Return ONLY valid JSON. "
        "Never invent customer codes, port codes, quote refs, or booking refs. "
        "Use null for missing values. Keys: customer_code, origin, destination, "
        "departure_after, sailing_id, voyage_number, container_type, container_qty, "
        "cargo_weight_kg, cargo_description, dangerous_goods, quote_ref, booking_ref, "
        "entity, query_mode, status, limit, reference. query_mode must be count, "
        "summary, or list. entity must be customers, ports, vessels, sailings, "
        "quotations, bookings, containers, shipment_events, or all. "
        f"Requested action: {action}. Reference data: {json.dumps(refs)}"
    )
    response = get_llm(temperature=0, reasoning=False).invoke(
        [SystemMessage(content=system), HumanMessage(content=prompt)]
    )
    return _json_object(str(response.content))


def _port_from_text(text: str, refs: dict[str, Any]) -> list[str]:
    lower = text.lower()
    known_codes = {port["unlocode"].upper() for port in refs["ports"]}
    found: list[tuple[int, str]] = []
    for port in refs["ports"]:
        code = port["unlocode"].upper()
        for needle in (code.lower(), port["name"].lower()):
            match = re.search(rf"\b{re.escape(needle)}\b", lower)
            if match:
                found.append((match.start(), code))
    for alias, code in sorted(_PORT_ALIASES.items(), key=lambda item: -len(item[0])):
        if code not in known_codes:
            continue
        match = re.search(rf"\b{re.escape(alias)}\b", lower)
        if match:
            found.append((match.start(), code))
    ordered: list[str] = []
    for _, code in sorted(found):
        if code not in ordered:
            ordered.append(code)
    return ordered


def _known_port_codes(refs: dict[str, Any]) -> set[str]:
    return {port["unlocode"].upper() for port in refs["ports"]}


def _sanitize_params(params: dict[str, Any], refs: dict[str, Any]) -> dict[str, Any]:
    known_ports = _known_port_codes(refs)
    clean: dict[str, Any] = {}
    for key, value in params.items():
        if key not in _PARAM_KEYS or value in (None, "", [], {}):
            continue
        if key in {"origin", "destination"}:
            code = str(value).upper().strip()
            alias = _PORT_ALIASES.get(code.lower())
            if alias:
                code = alias
            if code not in known_ports:
                continue
            clean[key] = code
            continue
        if key == "sailing_id":
            try:
                clean[key] = int(value)
            except (TypeError, ValueError):
                voyage = str(value).upper().strip()
                if re.fullmatch(r"[A-Z]{2}\d{3}[A-Z]", voyage):
                    clean["voyage_number"] = voyage
            continue
        if key == "voyage_number":
            voyage = str(value).upper().strip()
            if re.fullmatch(r"[A-Z]{2}\d{3}[A-Z]", voyage):
                clean[key] = voyage
            continue
        if key == "status" and str(value).lower() not in _KNOWN_STATUSES:
            continue
        if key == "entity":
            entity = str(value).lower().strip()
            if entity == "all" or entity in _ENTITY_TERMS:
                clean[key] = entity
            continue
        if key == "query_mode" and str(value).lower() not in {
            "count",
            "summary",
            "list",
        }:
            continue
        if key == "reference":
            token = str(value).strip().upper()
            if token in {
                "AMOUNT",
                "amounts",
                "total",
                "totals",
                "sum",
                "value",
                "values",
                "price",
                "prices",
                "list",
                "last",
            }:
                continue
            clean[key] = token
            continue
        if key == "include_amounts":
            clean[key] = bool(value)
            continue
        clean[key] = value
    return clean


def _fallback_extract(prompt: str, action: str) -> dict[str, Any]:
    marker = "Additional information from the user:"
    if marker in prompt:
        original, follow_up = prompt.rsplit(marker, 1)
        merged = _fallback_extract(original.strip(), action)
        updates = _fallback_extract(follow_up.strip(), action)
        if not re.search(
            r"\b(DG|dangerous goods?|hazardous)\b",
            follow_up,
            re.IGNORECASE,
        ):
            updates.pop("dangerous_goods", None)
        if not re.search(
            r"\b(general merchandise|general cargo|electronics|cargo)\b",
            follow_up,
            re.IGNORECASE,
        ):
            updates.pop("cargo_description", None)
        if updates.get("origin"):
            merged.pop("unrecognized_origin", None)
        if updates.get("destination"):
            merged.pop("unrecognized_destination", None)
        merged.update(updates)
        return merged

    refs = repository.list_reference_data()
    upper = prompt.upper()
    dangerous_goods_denied = bool(
        re.search(
            r"\b(no|not|without|non[-\s]?)\s*(dangerous goods?|hazardous|DG)\b",
            prompt,
            re.IGNORECASE,
        )
    )
    params: dict[str, Any] = {
        "dangerous_goods": not dangerous_goods_denied and bool(
            re.search(r"\b(DG|dangerous goods?|hazardous)\b", prompt, re.IGNORECASE)
        )
    }
    for customer in refs["customers"]:
        if customer["customer_code"].upper() in upper:
            params["customer_code"] = customer["customer_code"]
            break
    if not params.get("customer_code"):
        # Unique customer name match, e.g. "ACME Industrial"
        for customer in refs["customers"]:
            name = str(customer.get("name") or "")
            if len(name) >= 4 and name.lower() in prompt.lower():
                resolved = resolve_customer(name)
                if resolved["status"] == "resolved":
                    params["customer_code"] = resolved["value"]
                    break
        code_match = re.search(
            r"\bcustomer(?:_code)?\s*[=:#-]?\s*([A-Z0-9-]{3,})\b",
            prompt,
            re.IGNORECASE,
        )
        if code_match:
            resolved = resolve_customer(code_match.group(1))
            if resolved["status"] == "resolved":
                params["customer_code"] = resolved["value"]
            elif resolved["status"] in {"unknown", "ambiguous"}:
                params["customer_code"] = code_match.group(1).upper()

    # Structured choice / patch follow-ups: "customer_code: ACME-IN, origin: USMIA"
    if re.search(r"\b[a-z_]+\s*:\s*\S+", prompt, re.IGNORECASE):
        patch_values = parse_choice_value(prompt)
        if patch_values:
            params.update(patch_values)

    ports = _port_from_text(prompt, refs)
    if ports:
        params["origin"] = ports[0]
    if len(ports) > 1:
        params["destination"] = ports[1]
    coded_route = re.search(
        r"\bfrom\s+([A-Z]{3,5})\s+to\s+([A-Z]{3,5})\b",
        prompt,
        re.IGNORECASE,
    )
    if coded_route:
        for role, token in zip(
            ("origin", "destination"),
            coded_route.groups(),
        ):
            resolved = resolve_port(token)
            if resolved["status"] == "resolved":
                params[role] = resolved["value"]
                params.pop(f"unrecognized_{role}", None)
            else:
                params.pop(role, None)
                params[f"unrecognized_{role}"] = token.upper()

    container = re.search(
        r"\b(?:(\d+)\s*[xX]\s*)?(20GP|40GP|40HC|40RF)\b",
        prompt,
        re.IGNORECASE,
    )
    if container:
        params["container_qty"] = int(container.group(1) or 1)
        params["container_type"] = container.group(2).upper()
    weight = re.search(
        r"\b(\d[\d,]*(?:\.\d+)?)\s*(kg|tonnes?|tons?|mt)\b",
        prompt,
        re.IGNORECASE,
    )
    if weight:
        value = float(weight.group(1).replace(",", ""))
        if weight.group(2).lower() != "kg":
            value *= 1000
        params["cargo_weight_kg"] = value
    sailing = re.search(r"\bsailing(?:_id)?\s*[:#]?\s*(\d+)\b", prompt, re.IGNORECASE)
    if sailing:
        params["sailing_id"] = int(sailing.group(1))
    voyage = re.search(r"\b([A-Z]{2}\d{3}[A-Z])\b", upper)
    if voyage:
        params["voyage_number"] = voyage.group(1)
    quote = re.search(r"\bSLQ-[A-Z0-9]+\b", upper)
    if quote:
        params["quote_ref"] = quote.group(0)
    booking = re.search(r"\bSLB-[A-Z0-9]+\b", upper)
    if booking:
        params["booking_ref"] = booking.group(0)
    departure = re.search(r"\b20\d{2}-\d{2}-\d{2}\b", prompt)
    if departure:
        params["departure_after"] = departure.group(0)
    if action == "create_quotation":
        params.setdefault("cargo_description", "General cargo")
    if action == "data_query":
        lower_prompt = prompt.lower()
        wants_records = bool(
            re.search(
                r"\b(list|show|display|latest|last|recent|newest|oldest|fetch|"
                r"amount|amounts|total|totals|sum|calculate|calc|"
                r"detail|details|more detail|more details|tell me more|elaborate)\b",
                lower_prompt,
            )
        )
        wants_amounts = bool(
            re.search(
                r"\b(amount|amounts|total(?:s)?(?:\s+usd)?|sum|value|values|"
                r"price|prices)\b",
                lower_prompt,
            )
        )
        params["query_mode"] = (
            "summary"
            if any(word in lower_prompt for word in ("summary", "by status"))
            and not wants_amounts
            else (
                "list"
                if wants_records or wants_amounts
                else (
                    "count"
                    if any(word in lower_prompt for word in ("how many", "count"))
                    else "list"
                )
            )
        )
        if "overview" in prompt.lower() or "all data" in prompt.lower():
            params["entity"] = "all"
            params["query_mode"] = "summary"
        else:
            for entity, terms in _ENTITY_TERMS.items():
                if any(
                    re.search(rf"\b{re.escape(term)}\b", prompt, re.IGNORECASE)
                    for term in terms
                ):
                    params["entity"] = entity
                    break
            if not params.get("entity") and re.search(
                r"\bsailing\b", prompt, re.IGNORECASE
            ):
                params["entity"] = "sailings"
            # Amount/total prompts without an explicit entity imply quotations.
            if not params.get("entity") and wants_amounts:
                params["entity"] = "quotations"
        for status in _KNOWN_STATUSES:
            if re.search(rf"\b{re.escape(status)}\b", prompt, re.IGNORECASE):
                params["status"] = status
                break
        requested_limit = re.search(
            r"\b(?:top|latest|recent|first|last|limit|newest)\s+(\d+)\b",
            prompt,
            re.IGNORECASE,
        )
        if not requested_limit:
            requested_limit = re.search(
                r"\b(\d+)\s+(?:latest|recent|last|newest)?\s*"
                r"(?:quotation|quotations|quote|quotes|booking|bookings|"
                r"sailing|sailings|amount|amounts|total|totals)\b",
                prompt,
                re.IGNORECASE,
            )
        if requested_limit:
            params["limit"] = min(max(int(requested_limit.group(1)), 1), 25)
        elif re.search(
            r"\b(last|latest|newest|most recent)\s+"
            r"(quotation|quote|booking|sailing|customer|container|event)\b",
            lower_prompt,
        ):
            params["limit"] = 1
        params.setdefault("limit", 10)
        if wants_amounts:
            params["query_mode"] = "list"
            params["include_amounts"] = True
    return _sanitize_params(params, refs)


def _is_complete(action: str, params: dict[str, Any]) -> bool:
    if action == "create_quotation":
        has_sailing = bool(
            params.get("sailing_id")
            or params.get("voyage_number")
            or (params.get("origin") and params.get("destination"))
        )
        return has_sailing and all(
            params.get(key) not in (None, "")
            for key in (
                "customer_code",
                "container_type",
                "container_qty",
                "cargo_weight_kg",
            )
        )
    required = {
        "conversation": (),
        "reference_data": (),
        "data_query": ("entity", "query_mode"),
        "get_quotation": ("quote_ref",),
        "search_sailings": ("origin", "destination"),
        "track_booking": ("booking_ref",),
        "create_booking": ("quote_ref",),
        "create_quotation": (),
    }[action]
    return all(params.get(key) not in (None, "") for key in required)


def extract_parameters(
    prompt: str,
    action: str,
    *,
    patches: dict[str, Any] | None = None,
) -> dict[str, Any]:
    refs = repository.list_reference_data()
    fallback = _fallback_extract(prompt, action)
    if patches:
        fallback = apply_patches(fallback, patches)
    # Most UI/example prompts contain explicit codes and quantities. Avoid an
    # expensive LLM round-trip when deterministic parsing already has enough.
    if _is_complete(action, fallback):
        return fallback
    try:
        extracted = _llm_extract(prompt, action)
    except Exception:  # noqa: BLE001
        extracted = {}
    valid = _sanitize_params(
        {
            key: value
            for key, value in extracted.items()
            if value not in (None, "", [], {})
        },
        refs,
    )
    # Deterministic values win for IDs/codes explicitly found in the prompt.
    valid.update(fallback)
    for role in ("origin", "destination"):
        if valid.get(f"unrecognized_{role}"):
            valid.pop(role, None)
    return valid
