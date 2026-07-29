"""Prompt-to-parameters extraction with deterministic fallback."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from Learning.llm import get_llm
from projects.shipping_logistics_agent import repository

_PORT_ALIASES = {
    "mumbai": "INNSA",
    "nhava sheva": "INNSA",
    "jnpt": "INNSA",
    "singapore": "SGSIN",
    "jebel ali": "AEDXB",
    "dubai": "AEDXB",
    "rotterdam": "NLRTM",
    "new york": "USNYC",
    "nyc": "USNYC",
    "miami": "USMIA",
    "mia": "USMIA",
    "london": "GBLGP",
    "lon": "GBLGP",
    "london gateway": "GBLGP",
}
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
    response = get_llm(temperature=0, reasoning=False).invoke(
        [
            SystemMessage(
                content=(
                    "Classify the user's shipping/logistics request. Return ONLY "
                    "JSON with an action key. Allowed actions: conversation, "
                    "reference_data, data_query, get_quotation, search_sailings, "
                    "track_booking, create_quotation, create_booking. Creating a "
                    "quotation or booking is a write "
                    "request. Questions about routes, vessels, schedules, departure "
                    "or arrival are search_sailings. Shipment status is "
                    "track_booking. General capability, customer, port, or container "
                    "questions are reference_data. Counts, summaries, recent records, "
                    "or lists of customers, ports, vessels, sailings, quotations, "
                    "bookings, containers, or shipment events are data_query. Asking "
                    "what a sailing id is, or to show/list sailings without a route, "
                    "is data_query. Read verbs such as count, list, show, latest, "
                    "last, recent, and existing must never create data, even when "
                    "the prompt contains quotation or booking. Only explicit create, "
                    "prepare, generate, make, or confirm intent is a write. Looking "
                    "up one SLQ reference is get_quotation. "
                    "Greetings, thanks, and casual conversation are conversation."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )
    action = _json_object(str(response.content)).get("action")
    return action if action in _ACTIONS else "reference_data"


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
            r"recent|latest|last|newest|oldest|history|all|existing|current)\b",
            text,
            re.IGNORECASE,
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


def classify_action(prompt: str) -> str:
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
    if quotation_terms and _has_read_intent(text):
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
    if "book" in text or "booking" in text:
        return "create_booking"
    if any(word in text for word in ("port", "customer", "container type", "help")):
        return "reference_data"
    try:
        return _llm_classify_action(prompt)
    except Exception:  # noqa: BLE001
        return "reference_data"


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
        known_ports = _known_port_codes(refs)
        for role, token in zip(
            ("origin", "destination"),
            coded_route.groups(),
        ):
            code = token.upper()
            code = _PORT_ALIASES.get(code.lower(), code)
            if code in known_ports:
                params[role] = code
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
                r"\b(list|show|display|latest|last|recent|newest|oldest|fetch)\b",
                lower_prompt,
            )
        )
        params["query_mode"] = (
            "summary"
            if any(word in lower_prompt for word in ("summary", "by status"))
            else (
                "list"
                if wants_records
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
        for status in _KNOWN_STATUSES:
            if re.search(rf"\b{re.escape(status)}\b", prompt, re.IGNORECASE):
                params["status"] = status
                break
        requested_limit = re.search(
            r"\b(?:top|latest|recent|first|limit)\s+(\d+)\b",
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


def extract_parameters(prompt: str, action: str) -> dict[str, Any]:
    refs = repository.list_reference_data()
    fallback = _fallback_extract(prompt, action)
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
