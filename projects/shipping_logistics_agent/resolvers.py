"""Unique-match resolvers and structured recovery helpers for shipping intents.

Policy: auto-fill only when exactly one candidate matches. Ambiguous or missing
values become selectable recovery choices; never silently pick among multiples.
"""

from __future__ import annotations

import re
from typing import Any

from projects.shipping_logistics_agent import repository

PORT_ALIASES = {
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
    "atlanta": "ATL",  # may be unsupported — kept for explicit unknown reporting
}

CONTAINER_TYPES = sorted(repository.ALLOWED_CONTAINER_TYPES)
DATA_ENTITIES = sorted(repository.DATA_ENTITIES)


def _choice(
    kind: str,
    field: str,
    label: str,
    value: str,
    *,
    reason: str = "",
) -> dict[str, str]:
    return {
        "kind": kind,
        "field": field,
        "label": label,
        "value": value,
        "reason": reason,
    }


def resolve_customer(token: str | None) -> dict[str, Any]:
    """Resolve a customer by exact code or unique name substring."""
    if not token:
        return {"status": "missing", "value": None, "candidates": []}
    refs = repository.list_reference_data()["customers"]
    needle = token.strip().upper()
    exact = [c for c in refs if c["customer_code"].upper() == needle]
    if len(exact) == 1:
        return {"status": "resolved", "value": exact[0]["customer_code"], "candidates": exact}
    name_hits = [
        c
        for c in refs
        if needle in str(c.get("name") or "").upper()
        or needle in str(c.get("customer_code") or "").upper()
    ]
    if len(name_hits) == 1:
        return {
            "status": "resolved",
            "value": name_hits[0]["customer_code"],
            "candidates": name_hits,
        }
    if name_hits:
        return {"status": "ambiguous", "value": None, "candidates": name_hits}
    return {"status": "unknown", "value": None, "candidates": refs}


def resolve_port(token: str | None) -> dict[str, Any]:
    """Resolve a port by UN/LOCODE, alias, or unique name match."""
    if not token:
        return {"status": "missing", "value": None, "candidates": []}
    refs = repository.list_reference_data()["ports"]
    known = {p["unlocode"].upper(): p for p in refs}
    raw = token.strip()
    upper = raw.upper()
    alias = PORT_ALIASES.get(raw.lower())
    if alias and alias in known:
        return {"status": "resolved", "value": alias, "candidates": [known[alias]]}
    if upper in known:
        return {"status": "resolved", "value": upper, "candidates": [known[upper]]}
    name_hits = [
        p
        for p in refs
        if raw.lower() in str(p.get("name") or "").lower()
        or upper in str(p.get("unlocode") or "").upper()
    ]
    if len(name_hits) == 1:
        return {
            "status": "resolved",
            "value": name_hits[0]["unlocode"].upper(),
            "candidates": name_hits,
        }
    if name_hits:
        return {"status": "ambiguous", "value": None, "candidates": name_hits}
    return {"status": "unknown", "value": None, "candidates": refs, "token": upper}


def resolve_voyage(token: str | None, *, origin: str | None = None, destination: str | None = None) -> dict[str, Any]:
    """Resolve a voyage number to a scheduled sailing when unique."""
    if not token:
        return {"status": "missing", "value": None, "candidates": []}
    voyage = token.strip().upper()
    sailings = (
        repository.query_shipping_data(
            "sailings",
            operation="list",
            status="scheduled",
            limit=25,
        ).get("records")
        or []
    )
    hits = [s for s in sailings if str(s.get("voyage_number") or "").upper() == voyage]
    if origin:
        hits = [s for s in hits if s.get("origin") == origin]
    if destination:
        hits = [s for s in hits if s.get("destination") == destination]
    if len(hits) == 1:
        return {"status": "resolved", "value": hits[0], "candidates": hits}
    if hits:
        return {"status": "ambiguous", "value": None, "candidates": hits}
    # Fall back to catalog for choices
    return {"status": "unknown", "value": None, "candidates": sailings, "token": voyage}


def resolve_sailing_id(sailing_id: int | None) -> dict[str, Any]:
    if sailing_id is None:
        return {"status": "missing", "value": None, "candidates": []}
    sailings = (
        repository.query_shipping_data(
            "sailings",
            operation="list",
            status="scheduled",
            limit=25,
        ).get("records")
        or []
    )
    hits = [s for s in sailings if s.get("sailing_id") == sailing_id]
    if len(hits) == 1:
        return {"status": "resolved", "value": hits[0], "candidates": hits}
    return {"status": "unknown", "value": None, "candidates": sailings, "token": sailing_id}


def resolve_quote_ref(token: str | None) -> dict[str, Any]:
    if not token:
        return {"status": "missing", "value": None, "candidates": []}
    quote = repository.get_quotation(token.strip().upper())
    if quote:
        return {"status": "resolved", "value": quote["quote_ref"], "candidates": [quote]}
    recent = (
        repository.query_shipping_data(
            "quotations",
            operation="list",
            limit=8,
        ).get("records")
        or []
    )
    return {"status": "unknown", "value": None, "candidates": recent, "token": token}


def resolve_booking_ref(token: str | None) -> dict[str, Any]:
    if not token:
        return {"status": "missing", "value": None, "candidates": []}
    booking = repository.track_booking(token.strip().upper())
    if booking:
        return {
            "status": "resolved",
            "value": booking["booking_ref"],
            "candidates": [booking],
        }
    recent = (
        repository.query_shipping_data(
            "bookings",
            operation="list",
            limit=8,
        ).get("records")
        or []
    )
    return {"status": "unknown", "value": None, "candidates": recent, "token": token}


def resolve_entity(token: str | None) -> dict[str, Any]:
    if not token:
        return {"status": "missing", "value": None, "candidates": DATA_ENTITIES}
    canonical = repository.ENTITY_ALIASES.get(token.lower(), token.lower())
    if canonical in repository.DATA_ENTITIES or canonical == "all":
        return {"status": "resolved", "value": canonical, "candidates": [canonical]}
    return {"status": "unknown", "value": None, "candidates": DATA_ENTITIES, "token": token}


def resolve_container_type(token: str | None) -> dict[str, Any]:
    if not token:
        return {"status": "missing", "value": None, "candidates": CONTAINER_TYPES}
    upper = token.strip().upper()
    if upper in repository.ALLOWED_CONTAINER_TYPES:
        return {"status": "resolved", "value": upper, "candidates": [upper]}
    return {"status": "unknown", "value": None, "candidates": CONTAINER_TYPES, "token": upper}


def customer_choices(reason: str = "Select a valid customer") -> list[dict[str, str]]:
    return [
        _choice(
            "customer",
            "customer_code",
            f"{c['customer_code']} — {c['name']}",
            f"customer_code: {c['customer_code']}",
            reason=reason,
        )
        for c in repository.list_reference_data()["customers"]
    ]


def port_choices(field: str, reason: str = "Select a valid port") -> list[dict[str, str]]:
    return [
        _choice(
            "port",
            field,
            f"{p['unlocode']} — {p['name']}",
            f"{field}: {p['unlocode']}",
            reason=reason,
        )
        for p in repository.list_reference_data()["ports"]
    ]


def sailing_choices(
    *,
    origin: str | None = None,
    destination: str | None = None,
    reason: str = "Select a valid sailing",
    limit: int = 8,
) -> list[dict[str, str]]:
    sailings = (
        repository.query_shipping_data(
            "sailings",
            operation="list",
            status="scheduled",
            limit=25,
        ).get("records")
        or []
    )
    filtered = [
        s
        for s in sailings
        if (not origin or s.get("origin") == origin)
        and (not destination or s.get("destination") == destination)
    ] or sailings
    choices: list[dict[str, str]] = []
    for sailing in filtered[:limit]:
        voyage = str(sailing.get("voyage_number"))
        origin_code = str(sailing.get("origin"))
        destination_code = str(sailing.get("destination"))
        choices.append(
            _choice(
                "sailing",
                "voyage_number",
                f"{voyage} — {origin_code} → {destination_code}",
                (
                    f"voyage_number: {voyage}, "
                    f"origin: {origin_code}, "
                    f"destination: {destination_code}"
                ),
                reason=reason,
            )
        )
    return choices


def container_type_choices(reason: str = "Select a container type") -> list[dict[str, str]]:
    return [
        _choice(
            "container_type",
            "container_type",
            kind,
            f"container_type: {kind}",
            reason=reason,
        )
        for kind in CONTAINER_TYPES
    ]


def entity_choices(reason: str = "Select a data entity") -> list[dict[str, str]]:
    return [
        _choice(
            "entity",
            "entity",
            entity,
            f"entity: {entity}",
            reason=reason,
        )
        for entity in DATA_ENTITIES
    ]


def quote_choices(reason: str = "Select a quotation") -> list[dict[str, str]]:
    records = (
        repository.query_shipping_data(
            "quotations",
            operation="list",
            limit=8,
        ).get("records")
        or []
    )
    return [
        _choice(
            "quote_ref",
            "quote_ref",
            (
                f"{r.get('quote_ref')} — {r.get('customer_code')} "
                f"{r.get('origin')}→{r.get('destination')}"
            ),
            f"quote_ref: {r.get('quote_ref')}",
            reason=reason,
        )
        for r in records
        if r.get("quote_ref")
    ]


def booking_choices(reason: str = "Select a booking") -> list[dict[str, str]]:
    records = (
        repository.query_shipping_data(
            "bookings",
            operation="list",
            limit=8,
        ).get("records")
        or []
    )
    return [
        _choice(
            "booking_ref",
            "booking_ref",
            (
                f"{r.get('booking_ref')} — {r.get('customer_code')} "
                f"{r.get('status')}"
            ),
            f"booking_ref: {r.get('booking_ref')}",
            reason=reason,
        )
        for r in records
        if r.get("booking_ref")
    ]


def status_choices(entity: str, reason: str = "Select a status") -> list[dict[str, str]]:
    statuses = sorted(repository.ENTITY_STATUSES.get(entity, set()))
    return [
        _choice(
            "status",
            "status",
            status,
            f"status: {status}",
            reason=reason,
        )
        for status in statuses
    ]


def apply_patches(base: dict[str, Any], patches: dict[str, Any]) -> dict[str, Any]:
    """Merge structured field patches into retained parameters."""
    merged = dict(base)
    for key, value in patches.items():
        if value in (None, "", [], {}):
            continue
        if key.startswith("clear_"):
            merged.pop(key.replace("clear_", "", 1), None)
            continue
        merged[key] = value
        if key == "origin":
            merged.pop("unrecognized_origin", None)
        if key == "destination":
            merged.pop("unrecognized_destination", None)
    return merged


def parse_choice_value(value: str) -> dict[str, Any]:
    """Parse a choice value like 'customer_code: ACME-IN, origin: USMIA'."""
    patches: dict[str, Any] = {}
    for part in re.split(r",\s*", value.strip()):
        if ":" not in part:
            continue
        key, raw = part.split(":", 1)
        key = key.strip()
        raw = raw.strip()
        if key == "sailing_id":
            try:
                patches[key] = int(raw)
            except ValueError:
                patches["voyage_number"] = raw.upper()
        elif key in {"container_qty", "limit"}:
            try:
                patches[key] = int(raw)
            except ValueError:
                continue
        elif key == "cargo_weight_kg":
            try:
                patches[key] = float(raw.replace(",", ""))
            except ValueError:
                continue
        elif key == "dangerous_goods":
            patches[key] = raw.lower() in {"true", "1", "yes"}
        else:
            patches[key] = raw
    return patches


def build_recovery(
    action: str,
    params: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    """Build a structured recovery payload for any invalid shipping request."""
    error_text = " ".join(errors).lower()
    missing: list[str] = []
    invalid: list[str] = []
    choices: list[dict[str, str]] = []
    groups: list[dict[str, Any]] = []

    def add_group(field: str, title: str, group_choices: list[dict[str, str]]) -> None:
        if not group_choices:
            return
        groups.append(
            {
                "field": field,
                "title": title,
                "choices": group_choices,
            }
        )
        choices.extend(group_choices)

    # Parse missing keys from "Missing ...: a, b"
    for error in errors:
        if error.startswith("Missing "):
            detail = error.split(":", 1)[-1]
            for part in re.split(r"[(),]", detail):
                token = part.strip().lower().replace(" ", "_")
                if token in {
                    "customer_code",
                    "sailing_id",
                    "origin",
                    "destination",
                    "container_type",
                    "container_qty",
                    "cargo_weight_kg",
                    "quote_ref",
                    "booking_ref",
                    "entity",
                    "query_mode",
                }:
                    missing.append(token)
                if "entity" in token:
                    missing.append("entity")
                if "route" in token:
                    missing.extend(["origin", "destination"])

    if params.get("unrecognized_origin") or "unsupported origin" in error_text:
        invalid.append("origin")
        add_group(
            "origin",
            "Choose a valid origin port",
            port_choices("origin", "Unsupported origin"),
        )
    if params.get("unrecognized_destination") or "unsupported destination" in error_text:
        invalid.append("destination")
        add_group(
            "destination",
            "Choose a valid destination port",
            port_choices("destination", "Unsupported destination"),
        )

    if action == "create_quotation":
        customer = params.get("customer_code")
        if (
            "customer_code" in missing
            or not customer
            or "unknown customer" in error_text
            or "customer" in error_text
        ):
            if customer:
                invalid.append("customer_code")
            else:
                missing.append("customer_code")
            add_group(
                "customer_code",
                "Choose a customer",
                customer_choices("Valid sample customers"),
            )
        if (
            "sailing_id" in missing
            or "voyage" in error_text
            or "sailing" in error_text
            or "route" in error_text
            or not params.get("sailing_id")
        ):
            if params.get("voyage_number") or params.get("sailing_id"):
                invalid.append("voyage_number")
            else:
                missing.append("sailing_id")
            add_group(
                "voyage_number",
                "Choose a sailing / voyage",
                sailing_choices(
                    origin=params.get("origin"),
                    destination=params.get("destination"),
                    reason="Valid scheduled sailings",
                ),
            )
        if "container_type" in missing or (
            params.get("container_type")
            and params["container_type"] not in repository.ALLOWED_CONTAINER_TYPES
        ):
            missing.append("container_type")
            add_group(
                "container_type",
                "Choose a container type",
                container_type_choices(),
            )
        if "container_qty" in missing:
            missing.append("container_qty")
        if "cargo_weight_kg" in missing:
            missing.append("cargo_weight_kg")

    if action == "search_sailings":
        if "origin" in missing or not params.get("origin"):
            missing.append("origin")
            add_group("origin", "Choose origin", port_choices("origin"))
        if "destination" in missing or not params.get("destination"):
            missing.append("destination")
            add_group(
                "destination",
                "Choose destination",
                port_choices("destination"),
            )
        if "no scheduled sailing" in error_text or (
            params.get("origin") and params.get("destination")
        ):
            add_group(
                "voyage_number",
                "Available sailings",
                sailing_choices(
                    origin=params.get("origin"),
                    destination=params.get("destination"),
                ),
            )

    if action == "data_query":
        if "entity" in missing or not params.get("entity"):
            missing.append("entity")
            add_group("entity", "Choose what to query", entity_choices())
        entity = params.get("entity")
        if entity in repository.ENTITY_STATUSES and "status" in error_text:
            add_group(
                "status",
                "Choose a status filter",
                status_choices(str(entity)),
            )

    if action == "get_quotation":
        if "quote_ref" in missing or "quotation not found" in error_text:
            if params.get("quote_ref"):
                invalid.append("quote_ref")
            else:
                missing.append("quote_ref")
            add_group("quote_ref", "Choose a quotation", quote_choices())

    if action == "create_booking":
        if "quote_ref" in missing or "quotation" in error_text:
            missing.append("quote_ref")
            add_group(
                "quote_ref",
                "Choose an approved quotation to book",
                quote_choices("Bookable quotations"),
            )

    if action == "track_booking":
        if "booking_ref" in missing or "booking not found" in error_text:
            if params.get("booking_ref"):
                invalid.append("booking_ref")
            else:
                missing.append("booking_ref")
            add_group("booking_ref", "Choose a booking", booking_choices())

    # Deduplicate missing/invalid
    missing = list(dict.fromkeys(missing))
    invalid = list(dict.fromkeys(invalid))

    filled = {
        key: value
        for key, value in params.items()
        if value not in (None, "", [], {})
        and not str(key).startswith("unrecognized_")
        and key not in {"selected_sailing"}
        and key not in invalid
        and key not in missing
    }

    return {
        "active": bool(missing or invalid or choices),
        "action": action,
        "filled": filled,
        "missing_fields": missing,
        "invalid_fields": invalid,
        "errors": errors,
        "groups": groups,
        "choices": choices,
        "message": (
            "I need a few valid details to continue. "
            "Pick an option below or type the missing fields."
            if (missing or invalid or choices)
            else ""
        ),
    }
