"""Safe, parameterized PostgreSQL operations for shipping tools."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from projects.shipping_logistics_agent import db
from projects.shipping_logistics_agent.config import ShippingConfig, load_config

ALLOWED_CONTAINER_TYPES = {"20GP", "40GP", "40HC", "40RF"}
DATA_ENTITIES = {
    "customers",
    "ports",
    "vessels",
    "sailings",
    "quotations",
    "bookings",
    "containers",
    "shipment_events",
}
ENTITY_ALIASES = {
    "customer": "customers",
    "port": "ports",
    "vessel": "vessels",
    "sailing": "sailings",
    "quotation": "quotations",
    "quote": "quotations",
    "quotes": "quotations",
    "booking": "bookings",
    "container": "containers",
    "event": "shipment_events",
    "events": "shipment_events",
    "shipment_event": "shipment_events",
}
ENTITY_STATUSES = {
    "customers": {"approved", "hold", "blocked"},
    "ports": {"active", "inactive"},
    "vessels": {"active", "inactive"},
    "sailings": {"scheduled", "departed", "arrived", "cancelled"},
    "quotations": {"approved", "accepted", "expired", "rejected"},
    "bookings": {
        "confirmed",
        "gate_in",
        "loaded",
        "departed",
        "arrived",
        "delivered",
        "cancelled",
    },
}

_DATA_SOURCES = {
    "customers": """
        SELECT c.customer_code AS reference, c.credit_status AS status,
               c.customer_code, NULL::text AS origin, NULL::text AS destination,
               c.created_at, c.name, c.country_code, c.email,
               c.credit_limit_usd
        FROM shipping.customers c
    """,
    "ports": """
        SELECT p.unlocode AS reference,
               CASE WHEN p.active THEN 'active' ELSE 'inactive' END AS status,
               NULL::text AS customer_code, p.unlocode AS origin,
               NULL::text AS destination, NULL::timestamptz AS created_at,
               p.name, p.country_code, p.timezone
        FROM shipping.ports p
    """,
    "vessels": """
        SELECT v.imo_number AS reference,
               CASE WHEN v.active THEN 'active' ELSE 'inactive' END AS status,
               NULL::text AS customer_code, NULL::text AS origin,
               NULL::text AS destination, NULL::timestamptz AS created_at,
               v.name, v.vessel_type, v.capacity_teu, v.flag_country
        FROM shipping.vessels v
    """,
    "sailings": """
        SELECT s.voyage_number AS reference, s.status,
               NULL::text AS customer_code, op.unlocode AS origin,
               dp.unlocode AS destination, s.created_at, s.id AS sailing_id,
               s.voyage_number, v.name AS vessel_name, v.imo_number,
               s.departure_at, s.arrival_at, s.available_teu,
               s.base_rate_20_usd, s.base_rate_40_usd,
               s.reefer_surcharge_usd, s.dangerous_goods_allowed
        FROM shipping.sailings s
        JOIN shipping.vessels v ON v.id = s.vessel_id
        JOIN shipping.ports op ON op.id = s.origin_port_id
        JOIN shipping.ports dp ON dp.id = s.destination_port_id
    """,
    "quotations": """
        SELECT q.quote_ref AS reference, q.status, c.customer_code,
               op.unlocode AS origin, dp.unlocode AS destination, q.created_at,
               q.quote_ref, c.name AS customer_name, s.voyage_number,
               q.container_type, q.container_qty, q.cargo_weight_kg,
               q.cargo_description, q.dangerous_goods, q.total_usd,
               q.valid_until
        FROM shipping.quotations q
        JOIN shipping.customers c ON c.id = q.customer_id
        JOIN shipping.sailings s ON s.id = q.sailing_id
        JOIN shipping.ports op ON op.id = s.origin_port_id
        JOIN shipping.ports dp ON dp.id = s.destination_port_id
    """,
    "bookings": """
        SELECT b.booking_ref AS reference, b.status, c.customer_code,
               op.unlocode AS origin, dp.unlocode AS destination, b.created_at,
               b.booking_ref, q.quote_ref, c.name AS customer_name,
               s.voyage_number, v.name AS vessel_name, s.departure_at,
               s.arrival_at
        FROM shipping.bookings b
        JOIN shipping.quotations q ON q.id = b.quotation_id
        JOIN shipping.customers c ON c.id = b.customer_id
        JOIN shipping.sailings s ON s.id = b.sailing_id
        JOIN shipping.vessels v ON v.id = s.vessel_id
        JOIN shipping.ports op ON op.id = s.origin_port_id
        JOIN shipping.ports dp ON dp.id = s.destination_port_id
    """,
    "containers": """
        SELECT ct.container_number AS reference, ct.status, c.customer_code,
               op.unlocode AS origin, dp.unlocode AS destination,
               b.created_at, ct.container_number, ct.container_type,
               ct.seal_number, ct.gross_weight_kg, b.booking_ref
        FROM shipping.containers ct
        LEFT JOIN shipping.bookings b ON b.id = ct.booking_id
        LEFT JOIN shipping.customers c ON c.id = b.customer_id
        LEFT JOIN shipping.sailings s ON s.id = b.sailing_id
        LEFT JOIN shipping.ports op ON op.id = s.origin_port_id
        LEFT JOIN shipping.ports dp ON dp.id = s.destination_port_id
    """,
    "shipment_events": """
        SELECT e.event_code AS reference, e.event_code AS status,
               c.customer_code, p.unlocode AS origin,
               NULL::text AS destination, e.event_time AS created_at,
               b.booking_ref, e.event_code, e.event_time,
               p.name AS port_name, e.description, e.source
        FROM shipping.shipment_events e
        JOIN shipping.bookings b ON b.id = e.booking_id
        JOIN shipping.customers c ON c.id = b.customer_id
        LEFT JOIN shipping.ports p ON p.id = e.port_id
    """,
}


def _json_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, UUID):
        return str(value)
    return value


def _record(columns: list[str], row: tuple[Any, ...]) -> dict[str, Any]:
    return {column: _json_value(value) for column, value in zip(columns, row)}


def list_reference_data() -> dict[str, Any]:
    with db.connect() as conn:
        customers = conn.execute(
            """
            SELECT customer_code, name, country_code, credit_status
            FROM shipping.customers
            ORDER BY customer_code
            """
        ).fetchall()
        ports = conn.execute(
            """
            SELECT unlocode, name, country_code
            FROM shipping.ports
            WHERE active
            ORDER BY unlocode
            """
        ).fetchall()
    return {
        "customers": [
            _record(
                ["customer_code", "name", "country_code", "credit_status"],
                row,
            )
            for row in customers
        ],
        "ports": [
            _record(["unlocode", "name", "country_code"], row)
            for row in ports
        ],
        "container_types": sorted(ALLOWED_CONTAINER_TYPES),
    }


def search_sailings(
    origin: str,
    destination: str,
    *,
    departure_after: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    safe_limit = min(max(int(limit), 1), 25)
    after = departure_after or datetime.now(timezone.utc).isoformat()
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT s.id, s.voyage_number, v.name, v.imo_number,
                   op.unlocode, op.name, dp.unlocode, dp.name,
                   s.departure_at, s.arrival_at, s.available_teu,
                   s.base_rate_20_usd, s.base_rate_40_usd,
                   s.reefer_surcharge_usd, s.dangerous_goods_allowed
            FROM shipping.sailings s
            JOIN shipping.vessels v ON v.id = s.vessel_id
            JOIN shipping.ports op ON op.id = s.origin_port_id
            JOIN shipping.ports dp ON dp.id = s.destination_port_id
            WHERE op.unlocode = %s
              AND dp.unlocode = %s
              AND s.departure_at >= %s::timestamptz
              AND s.status = 'scheduled'
            ORDER BY s.departure_at
            LIMIT %s
            """,
            (origin.upper(), destination.upper(), after, safe_limit),
        ).fetchall()
    columns = [
        "sailing_id",
        "voyage_number",
        "vessel_name",
        "imo_number",
        "origin_code",
        "origin_name",
        "destination_code",
        "destination_name",
        "departure_at",
        "arrival_at",
        "available_teu",
        "base_rate_20_usd",
        "base_rate_40_usd",
        "reefer_surcharge_usd",
        "dangerous_goods_allowed",
    ]
    return [_record(columns, row) for row in rows]


def track_booking(booking_ref: str) -> dict[str, Any]:
    with db.connect() as conn:
        booking = conn.execute(
            """
            SELECT b.booking_ref, b.status, q.quote_ref, c.customer_code,
                   s.voyage_number, v.name, op.unlocode, dp.unlocode,
                   s.departure_at, s.arrival_at
            FROM shipping.bookings b
            JOIN shipping.quotations q ON q.id = b.quotation_id
            JOIN shipping.customers c ON c.id = b.customer_id
            JOIN shipping.sailings s ON s.id = b.sailing_id
            JOIN shipping.vessels v ON v.id = s.vessel_id
            JOIN shipping.ports op ON op.id = s.origin_port_id
            JOIN shipping.ports dp ON dp.id = s.destination_port_id
            WHERE b.booking_ref = %s
            """,
            (booking_ref.upper(),),
        ).fetchone()
        if not booking:
            return {}
        events = conn.execute(
            """
            SELECT e.event_code, e.event_time, p.unlocode, e.description,
                   e.source
            FROM shipping.shipment_events e
            LEFT JOIN shipping.ports p ON p.id = e.port_id
            JOIN shipping.bookings b ON b.id = e.booking_id
            WHERE b.booking_ref = %s
            ORDER BY e.event_time DESC
            LIMIT 50
            """,
            (booking_ref.upper(),),
        ).fetchall()
    data = _record(
        [
            "booking_ref",
            "status",
            "quote_ref",
            "customer_code",
            "voyage_number",
            "vessel_name",
            "origin",
            "destination",
            "departure_at",
            "arrival_at",
        ],
        booking,
    )
    data["events"] = [
        _record(
            ["event_code", "event_time", "port", "description", "source"],
            row,
        )
        for row in events
    ]
    return data


def get_quotation(quote_ref: str) -> dict[str, Any]:
    with db.connect() as conn:
        row = conn.execute(
            """
            SELECT q.quote_ref, q.status, c.customer_code, c.name,
                   s.id, s.voyage_number, op.unlocode, dp.unlocode,
                   q.container_type, q.container_qty, q.cargo_weight_kg,
                   q.cargo_description, q.dangerous_goods,
                   q.ocean_freight_usd, q.surcharges_usd, q.total_usd,
                   q.valid_until, q.created_at
            FROM shipping.quotations q
            JOIN shipping.customers c ON c.id = q.customer_id
            JOIN shipping.sailings s ON s.id = q.sailing_id
            JOIN shipping.ports op ON op.id = s.origin_port_id
            JOIN shipping.ports dp ON dp.id = s.destination_port_id
            WHERE q.quote_ref = %s
            """,
            (quote_ref.upper(),),
        ).fetchone()
    if not row:
        return {}
    return _record(
        [
            "quote_ref",
            "status",
            "customer_code",
            "customer_name",
            "sailing_id",
            "voyage_number",
            "origin",
            "destination",
            "container_type",
            "container_qty",
            "cargo_weight_kg",
            "cargo_description",
            "dangerous_goods",
            "ocean_freight_usd",
            "surcharges_usd",
            "total_usd",
            "valid_until",
            "created_at",
        ],
        row,
    )


def query_shipping_data(
    entity: str,
    *,
    operation: str = "list",
    status: str | None = None,
    customer_code: str | None = None,
    origin: str | None = None,
    destination: str | None = None,
    reference: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Safely count, summarize, or list a bounded set of shipping records."""
    canonical = ENTITY_ALIASES.get(entity.lower(), entity.lower())
    if canonical not in DATA_ENTITIES:
        raise ValueError(f"entity must be one of {sorted(DATA_ENTITIES)}")
    mode = operation.lower()
    if mode not in {"count", "summary", "list"}:
        raise ValueError("operation must be count, summary, or list")
    normalized_status = status.lower() if status else None
    allowed_statuses = ENTITY_STATUSES.get(canonical)
    if normalized_status and allowed_statuses and normalized_status not in allowed_statuses:
        raise ValueError(
            f"status for {canonical} must be one of {sorted(allowed_statuses)}"
        )

    clauses: list[str] = []
    values: list[Any] = []
    for column, value in (
        ("status", normalized_status),
        ("customer_code", customer_code.upper() if customer_code else None),
        ("origin", origin.upper() if origin else None),
        ("destination", destination.upper() if destination else None),
        ("reference", reference.upper() if reference else None),
    ):
        if value:
            clauses.append(f"{column} = %s")
            values.append(value)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    source = _DATA_SOURCES[canonical]
    filters = {
        key: value
        for key, value in {
            "status": normalized_status,
            "customer_code": customer_code.upper() if customer_code else None,
            "origin": origin.upper() if origin else None,
            "destination": destination.upper() if destination else None,
            "reference": reference.upper() if reference else None,
        }.items()
        if value
    }

    with db.connect() as conn:
        if mode == "count":
            count = conn.execute(
                f"SELECT count(*) FROM ({source}) records {where}",  # noqa: S608
                tuple(values),
            ).fetchone()[0]
            return {
                "entity": canonical,
                "operation": mode,
                "filters": filters,
                "count": int(count),
            }
        if mode == "summary":
            rows = conn.execute(
                f"""
                SELECT status, count(*)
                FROM ({source}) records
                {where}
                GROUP BY status
                ORDER BY status
                """,  # noqa: S608
                tuple(values),
            ).fetchall()
            breakdown = {str(row[0]): int(row[1]) for row in rows}
            return {
                "entity": canonical,
                "operation": mode,
                "filters": filters,
                "count": sum(breakdown.values()),
                "by_status": breakdown,
            }

        safe_limit = min(max(int(limit), 1), 25)
        total_matching = int(
            conn.execute(
                f"SELECT count(*) FROM ({source}) records {where}",  # noqa: S608
                tuple(values),
            ).fetchone()[0]
        )
        cursor = conn.execute(
            f"""
            SELECT *
            FROM ({source}) records
            {where}
            ORDER BY created_at DESC NULLS LAST, reference
            LIMIT %s
            """,  # noqa: S608
            (*values, safe_limit),
        )
        rows = cursor.fetchall()
        columns = [column.name for column in cursor.description]
    records = [_record(columns, row) for row in rows]
    return {
        "entity": canonical,
        "operation": mode,
        "filters": filters,
        "total_matching": total_matching,
        "returned": len(records),
        "limit": safe_limit,
        "records": records,
    }


def shipping_data_overview() -> dict[str, Any]:
    """Return bounded aggregate counts without exposing arbitrary SQL."""
    entities: dict[str, Any] = {}
    for entity in sorted(DATA_ENTITIES):
        entities[entity] = query_shipping_data(entity, operation="summary")
    return {
        "entity": "all",
        "operation": "overview",
        "entity_count": len(entities),
        "total_records": sum(item["count"] for item in entities.values()),
        "entities": entities,
    }


def build_quotation_proposal(
    customer_code: str,
    sailing_id: int,
    container_type: str,
    container_qty: int,
    cargo_weight_kg: float,
    cargo_description: str,
    *,
    dangerous_goods: bool = False,
    config: ShippingConfig | None = None,
) -> dict[str, Any]:
    cfg = config or load_config()
    kind = container_type.upper()
    if kind not in ALLOWED_CONTAINER_TYPES:
        raise ValueError(
            f"container_type must be one of {sorted(ALLOWED_CONTAINER_TYPES)}"
        )
    if container_qty < 1 or container_qty > 100:
        raise ValueError("container_qty must be between 1 and 100")
    if cargo_weight_kg <= 0:
        raise ValueError("cargo_weight_kg must be positive")

    with db.connect() as conn:
        row = conn.execute(
            """
            SELECT c.id, c.customer_code, c.name, c.credit_status,
                   c.credit_limit_usd, s.id, s.voyage_number,
                   op.unlocode, dp.unlocode, s.departure_at, s.arrival_at,
                   s.available_teu, s.base_rate_20_usd,
                   s.base_rate_40_usd, s.reefer_surcharge_usd,
                   s.dangerous_goods_allowed
            FROM shipping.customers c
            CROSS JOIN shipping.sailings s
            JOIN shipping.ports op ON op.id = s.origin_port_id
            JOIN shipping.ports dp ON dp.id = s.destination_port_id
            WHERE c.customer_code = %s
              AND s.id = %s
              AND s.status = 'scheduled'
            """,
            (customer_code.upper(), sailing_id),
        ).fetchone()
    if not row:
        raise ValueError("Unknown customer or scheduled sailing")

    (
        customer_id,
        code,
        customer_name,
        credit_status,
        credit_limit,
        resolved_sailing_id,
        voyage_number,
        origin,
        destination,
        departure_at,
        arrival_at,
        available_teu,
        rate20,
        rate40,
        reefer_surcharge,
        dg_allowed,
    ) = row
    teu_per_container = 1 if kind == "20GP" else 2
    required_teu = container_qty * teu_per_container
    base_rate = rate20 if kind == "20GP" else rate40
    ocean_freight = Decimal(base_rate) * container_qty
    surcharges = ocean_freight * Decimal("0.12")
    if kind == "40RF":
        surcharges += Decimal(reefer_surcharge) * container_qty
    total = ocean_freight + surcharges
    warnings: list[str] = []
    if credit_status != "approved":
        warnings.append(f"Customer credit status is {credit_status}")
    if required_teu > available_teu:
        warnings.append(
            f"Capacity shortfall: needs {required_teu} TEU, "
            f"only {available_teu} available"
        )
    if dangerous_goods and not dg_allowed:
        warnings.append("Dangerous goods are not allowed on this sailing")
    if total > Decimal(credit_limit):
        warnings.append("Quotation exceeds customer credit limit")

    return {
        "action": "create_quotation",
        "customer_id": int(customer_id),
        "customer_code": code,
        "customer_name": customer_name,
        "credit_status": credit_status,
        "credit_limit_usd": float(credit_limit),
        "sailing_id": int(resolved_sailing_id),
        "voyage_number": voyage_number,
        "origin": origin,
        "destination": destination,
        "departure_at": _json_value(departure_at),
        "arrival_at": _json_value(arrival_at),
        "container_type": kind,
        "container_qty": int(container_qty),
        "required_teu": required_teu,
        "available_teu": int(available_teu),
        "cargo_weight_kg": float(cargo_weight_kg),
        "cargo_description": cargo_description.strip() or "General cargo",
        "dangerous_goods": bool(dangerous_goods),
        "dangerous_goods_allowed": bool(dg_allowed),
        "ocean_freight_usd": float(ocean_freight),
        "surcharges_usd": float(surcharges),
        "total_usd": float(total),
        "currency": cfg.currency,
        "valid_until": (
            datetime.now(timezone.utc).date()
            + timedelta(days=cfg.quote_valid_days)
        ).isoformat(),
        "warnings": warnings,
    }


def build_booking_proposal(quote_ref: str) -> dict[str, Any]:
    quotation = get_quotation(quote_ref)
    if not quotation:
        raise ValueError(f"Quotation not found: {quote_ref}")
    warnings: list[str] = []
    if quotation["status"] not in {"approved", "accepted"}:
        warnings.append(
            f"Quotation status is {quotation['status']}, not approved/accepted"
        )
    if date.fromisoformat(quotation["valid_until"]) < date.today():
        warnings.append("Quotation has expired")
    return {
        "action": "create_booking",
        "quote_ref": quotation["quote_ref"],
        "customer_code": quotation["customer_code"],
        "sailing_id": quotation["sailing_id"],
        "voyage_number": quotation["voyage_number"],
        "origin": quotation["origin"],
        "destination": quotation["destination"],
        "container_type": quotation["container_type"],
        "container_qty": quotation["container_qty"],
        "total_usd": quotation["total_usd"],
        "currency": "USD",
        "warnings": warnings,
    }


def create_approval_request(
    thread_id: str,
    action: str,
    proposal: dict[str, Any],
    risk_review: dict[str, Any],
) -> dict[str, Any]:
    approval_id = uuid4()
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO shipping.approval_requests
                (id, thread_id, action, proposal, risk_review, status)
            VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, 'pending')
            ON CONFLICT (thread_id) DO UPDATE SET
                action = EXCLUDED.action,
                proposal = EXCLUDED.proposal,
                risk_review = EXCLUDED.risk_review,
                status = 'pending',
                reviewer = NULL,
                reviewer_note = NULL,
                decided_at = NULL,
                executed_at = NULL
            """,
            (
                approval_id,
                thread_id,
                action,
                json.dumps(proposal),
                json.dumps(risk_review),
            ),
        )
        row = conn.execute(
            """
            SELECT id, status
            FROM shipping.approval_requests
            WHERE thread_id = %s
            """,
            (thread_id,),
        ).fetchone()
    return {
        "approval_id": str(row[0]),
        "thread_id": thread_id,
        "status": row[1],
        "action": action,
        "proposal": proposal,
        "risk_review": risk_review,
    }


def decide_approval(
    thread_id: str,
    *,
    approve: bool,
    reviewer: str,
    note: str = "",
) -> dict[str, Any]:
    status = "approved" if approve else "rejected"
    with db.connect() as conn:
        row = conn.execute(
            """
            UPDATE shipping.approval_requests
            SET status = %s, reviewer = %s, reviewer_note = %s,
                decided_at = now()
            WHERE thread_id = %s AND status = 'pending'
            RETURNING id, action, proposal, risk_review
            """,
            (status, reviewer, note, thread_id),
        ).fetchone()
    if not row:
        raise ValueError("Pending approval not found for thread")
    return {
        "approval_id": str(row[0]),
        "thread_id": thread_id,
        "status": status,
        "action": row[1],
        "proposal": row[2],
        "risk_review": row[3],
        "reviewer": reviewer,
        "note": note,
    }


def get_approval(thread_id: str) -> dict[str, Any]:
    with db.connect() as conn:
        row = conn.execute(
            """
            SELECT id, action, proposal, risk_review, status,
                   reviewer, reviewer_note, requested_at, decided_at
            FROM shipping.approval_requests
            WHERE thread_id = %s
            """,
            (thread_id,),
        ).fetchone()
    if not row:
        return {}
    return _record(
        [
            "approval_id",
            "action",
            "proposal",
            "risk_review",
            "status",
            "reviewer",
            "reviewer_note",
            "requested_at",
            "decided_at",
        ],
        row,
    )


def execute_approved(thread_id: str) -> dict[str, Any]:
    approval = get_approval(thread_id)
    if not approval:
        raise ValueError("Approval not found")
    if approval["status"] == "rejected":
        return {
            "executed": False,
            "status": "rejected",
            "message": "Human reviewer rejected the proposal; no write occurred.",
        }
    if approval["status"] != "approved":
        raise PermissionError("Human approval is required before execution")

    action = approval["action"]
    proposal = approval["proposal"]
    approval_id = approval["approval_id"]
    if action == "create_quotation":
        result = _insert_quotation(proposal, approval_id)
    elif action == "create_booking":
        result = _insert_booking(proposal, approval_id)
    else:
        raise ValueError(f"Unsupported approved action: {action}")

    with db.connect() as conn:
        conn.execute(
            """
            UPDATE shipping.approval_requests
            SET status = 'executed', executed_at = now()
            WHERE thread_id = %s
            """,
            (thread_id,),
        )
        conn.execute(
            """
            INSERT INTO shipping.audit_log
                (thread_id, actor, action, entity_type, entity_id, payload)
            VALUES (%s, 'multi_agent', %s, %s, %s, %s::jsonb)
            """,
            (
                thread_id,
                action,
                result["entity_type"],
                result["entity_id"],
                json.dumps(result),
            ),
        )
    return {"executed": True, "status": "executed", **result}


def _insert_quotation(
    proposal: dict[str, Any],
    approval_id: str,
) -> dict[str, Any]:
    quote_id = uuid4()
    quote_ref = f"SLQ-{quote_id.hex[:10].upper()}"
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO shipping.quotations (
                id, quote_ref, customer_id, sailing_id, container_type,
                container_qty, cargo_weight_kg, cargo_description,
                dangerous_goods, ocean_freight_usd, surcharges_usd,
                total_usd, status, valid_until, approval_id
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, 'approved', %s::date, %s
            )
            """,
            (
                quote_id,
                quote_ref,
                proposal["customer_id"],
                proposal["sailing_id"],
                proposal["container_type"],
                proposal["container_qty"],
                proposal["cargo_weight_kg"],
                proposal["cargo_description"],
                proposal["dangerous_goods"],
                proposal["ocean_freight_usd"],
                proposal["surcharges_usd"],
                proposal["total_usd"],
                proposal["valid_until"],
                approval_id,
            ),
        )
    return {
        "entity_type": "quotation",
        "entity_id": str(quote_id),
        "quote_ref": quote_ref,
        "total_usd": proposal["total_usd"],
        "valid_until": proposal["valid_until"],
    }


def _insert_booking(
    proposal: dict[str, Any],
    approval_id: str,
) -> dict[str, Any]:
    booking_id = uuid4()
    booking_ref = f"SLB-{booking_id.hex[:10].upper()}"
    with db.connect() as conn:
        quote = conn.execute(
            """
            SELECT q.id, q.customer_id, q.sailing_id, q.container_type,
                   q.container_qty, s.available_teu
            FROM shipping.quotations q
            JOIN shipping.sailings s ON s.id = q.sailing_id
            WHERE q.quote_ref = %s
              AND q.status IN ('approved', 'accepted')
              AND q.valid_until >= current_date
            FOR UPDATE OF q, s
            """,
            (proposal["quote_ref"],),
        ).fetchone()
        if not quote:
            raise ValueError("Quotation is not valid for booking")
        quote_id, customer_id, sailing_id, container_type, qty, available = quote
        required_teu = qty * (1 if container_type == "20GP" else 2)
        if required_teu > available:
            raise ValueError("Insufficient sailing capacity")
        conn.execute(
            """
            INSERT INTO shipping.bookings (
                id, booking_ref, quotation_id, customer_id, sailing_id,
                status, approval_id
            ) VALUES (%s, %s, %s, %s, %s, 'confirmed', %s)
            """,
            (
                booking_id,
                booking_ref,
                quote_id,
                customer_id,
                sailing_id,
                approval_id,
            ),
        )
        conn.execute(
            """
            UPDATE shipping.sailings
            SET available_teu = available_teu - %s
            WHERE id = %s
            """,
            (required_teu, sailing_id),
        )
        origin_id = conn.execute(
            "SELECT origin_port_id FROM shipping.sailings WHERE id = %s",
            (sailing_id,),
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO shipping.shipment_events (
                booking_id, event_code, event_time, port_id, description
            ) VALUES (
                %s, 'BOOKING_CONFIRMED', now(), %s,
                'Booking confirmed after human approval'
            )
            """,
            (booking_id, origin_id),
        )
    return {
        "entity_type": "booking",
        "entity_id": str(booking_id),
        "booking_ref": booking_ref,
        "quote_ref": proposal["quote_ref"],
        "status": "confirmed",
    }

