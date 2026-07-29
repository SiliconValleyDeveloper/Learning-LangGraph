"""PostgreSQL-backed shipping MCP server.

All business writes require an approved ``shipping.approval_requests`` row.
There is intentionally no generic SQL execution tool.
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from projects.shipping_logistics_agent import repository
from projects.shipping_logistics_agent.rag import policy_candidates, rerank_evidence

mcp = FastMCP("shipping-logistics-postgres")


def _json(data: object) -> str:
    return json.dumps(data, indent=2, default=str)


@mcp.tool()
def shipping_reference_data() -> str:
    """List customer codes, ports, and supported container types."""
    return _json(repository.list_reference_data())


@mcp.tool()
def shipping_data_overview() -> str:
    """Count supported shipping entities and group each count by status."""
    return _json(repository.shipping_data_overview())


@mcp.tool()
def shipping_policy_knowledge(topic: str = "") -> str:
    """Retrieve ranked shipping approval, capacity, DG, pricing, or booking rules."""
    candidates = policy_candidates()
    if not topic.strip():
        return _json(candidates)
    return _json(rerank_evidence(topic, candidates, top_k=5))


@mcp.tool()
def query_shipping_data(
    entity: str,
    operation: str = "list",
    status: str = "",
    customer_code: str = "",
    origin: str = "",
    destination: str = "",
    reference: str = "",
    limit: int = 10,
) -> str:
    """Safely count, summarize, or list up to 25 filtered shipping records."""
    return _json(
        repository.query_shipping_data(
            entity,
            operation=operation,
            status=status or None,
            customer_code=customer_code or None,
            origin=origin or None,
            destination=destination or None,
            reference=reference or None,
            limit=limit,
        )
    )


@mcp.tool()
def search_sailings(
    origin: str,
    destination: str,
    departure_after: str = "",
    limit: int = 10,
) -> str:
    """Find scheduled sailings by UN/LOCODE (for example INNSA → SGSIN)."""
    return _json(
        repository.search_sailings(
            origin,
            destination,
            departure_after=departure_after or None,
            limit=limit,
        )
    )


@mcp.tool()
def calculate_quotation(
    customer_code: str,
    sailing_id: int,
    container_type: str,
    container_qty: int,
    cargo_weight_kg: float,
    cargo_description: str,
    dangerous_goods: bool = False,
) -> str:
    """Calculate a quotation proposal. This does not write a quotation."""
    return _json(
        repository.build_quotation_proposal(
            customer_code,
            sailing_id,
            container_type,
            container_qty,
            cargo_weight_kg,
            cargo_description,
            dangerous_goods=dangerous_goods,
        )
    )


@mcp.tool()
def get_quotation(quote_ref: str) -> str:
    """Return one quotation by reference."""
    return _json(repository.get_quotation(quote_ref))


@mcp.tool()
def track_booking(booking_ref: str) -> str:
    """Return booking status, voyage details, and shipment events."""
    return _json(repository.track_booking(booking_ref))


@mcp.tool()
def execute_human_approved_action(thread_id: str) -> str:
    """Execute a quotation/booking write only after recorded human approval."""
    return _json(repository.execute_approved(thread_id))


if __name__ == "__main__":
    mcp.run(transport="stdio")

