"""API consumer auth (F3.5) — hashed API keys, scopes, usage audit.

Keys are shown once at creation. Only SHA-256(pepper + key) is stored.
Pass key via ``X-API-Key`` or ``Authorization: Bearer <key>``.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import Depends, HTTPException, Request

from projects.finance_agent import db
from projects.finance_agent.config import FinanceConfig, load_config
from projects.finance_agent.logging_util import get_logger

log = get_logger("finance.auth")

TIER_DEFAULTS: dict[str, dict[str, Any]] = {
    "free": {
        "rate_limit_rpm": 60,
        "scopes": ["quotes:read"],
    },
    "pro": {
        "rate_limit_rpm": 300,
        "scopes": ["quotes:read", "quotes:refresh"],
    },
    "admin": {
        "rate_limit_rpm": 1000,
        "scopes": ["quotes:read", "quotes:refresh", "research:read", "admin"],
    },
}


@dataclass
class Consumer:
    id: int
    tenant_id: str
    name: str
    tier: str
    scopes: list[str]
    rate_limit_rpm: int
    is_active: bool

    def has_scope(self, scope: str) -> bool:
        if "admin" in self.scopes:
            return True
        return scope in self.scopes


def hash_api_key(raw_key: str, *, config: FinanceConfig | None = None) -> str:
    cfg = config or load_config()
    material = f"{cfg.api_key_pepper}:{raw_key}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def generate_api_key() -> str:
    return f"fk_{secrets.token_urlsafe(32)}"


def create_consumer(
    name: str,
    *,
    tenant_id: str = "default",
    tier: str | None = None,
    scopes: list[str] | None = None,
    rate_limit_rpm: int | None = None,
    config: FinanceConfig | None = None,
) -> dict[str, Any]:
    """Create consumer; returns dict including plaintext ``api_key`` (once)."""
    cfg = config or load_config()
    tier_name = (tier or cfg.api_default_tier).strip().lower()
    defaults = TIER_DEFAULTS.get(tier_name, TIER_DEFAULTS["free"])
    scope_list = scopes if scopes is not None else list(defaults["scopes"])
    rpm = rate_limit_rpm if rate_limit_rpm is not None else int(
        defaults.get("rate_limit_rpm") or cfg.api_default_rpm
    )
    raw_key = generate_api_key()
    key_hash = hash_api_key(raw_key, config=cfg)

    with db.connect(cfg) as conn:
        row = conn.execute(
            """
            INSERT INTO finance_api_consumers
                (tenant_id, name, api_key_hash, tier, scopes, rate_limit_rpm)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id, created_at
            """,
            (tenant_id, name, key_hash, tier_name, scope_list, rpm),
        ).fetchone()

    log.info(
        "consumer_created",
        extra={"consumer_id": row[0], "tier": tier_name, "consumer_name": name},
    )
    return {
        "id": int(row[0]),
        "tenant_id": tenant_id,
        "name": name,
        "tier": tier_name,
        "scopes": scope_list,
        "rate_limit_rpm": rpm,
        "api_key": raw_key,
        "created_at": str(row[1]),
        "note": "Store api_key now — it cannot be retrieved again.",
    }


def list_consumers(
    *,
    tenant_id: str | None = None,
    include_revoked: bool = False,
    config: FinanceConfig | None = None,
) -> list[dict[str, Any]]:
    cfg = config or load_config()
    clauses = ["1=1"]
    params: list[Any] = []
    if tenant_id:
        clauses.append("tenant_id = %s")
        params.append(tenant_id)
    if not include_revoked:
        clauses.append("is_active = TRUE AND revoked_at IS NULL")
    where = " AND ".join(clauses)
    with db.connect(cfg) as conn:
        rows = conn.execute(
            f"""
            SELECT id, tenant_id, name, tier, scopes, rate_limit_rpm,
                   is_active, created_at, revoked_at
            FROM finance_api_consumers
            WHERE {where}
            ORDER BY id
            """,
            params,
        ).fetchall()
    return [
        {
            "id": int(r[0]),
            "tenant_id": r[1],
            "name": r[2],
            "tier": r[3],
            "scopes": list(r[4] or []),
            "rate_limit_rpm": int(r[5]),
            "is_active": bool(r[6]),
            "created_at": str(r[7]),
            "revoked_at": str(r[8]) if r[8] else None,
        }
        for r in rows
    ]


def revoke_consumer(
    consumer_id: int,
    *,
    config: FinanceConfig | None = None,
) -> bool:
    cfg = config or load_config()
    with db.connect(cfg) as conn:
        row = conn.execute(
            """
            UPDATE finance_api_consumers
            SET is_active = FALSE, revoked_at = now(), updated_at = now()
            WHERE id = %s AND is_active = TRUE
            RETURNING id
            """,
            (consumer_id,),
        ).fetchone()
    ok = row is not None
    if ok:
        log.info("consumer_revoked", extra={"consumer_id": consumer_id})
    return ok


def lookup_consumer_by_key(
    raw_key: str,
    *,
    config: FinanceConfig | None = None,
) -> Consumer | None:
    cfg = config or load_config()
    key_hash = hash_api_key(raw_key, config=cfg)
    with db.connect(cfg) as conn:
        row = conn.execute(
            """
            SELECT id, tenant_id, name, tier, scopes, rate_limit_rpm, is_active
            FROM finance_api_consumers
            WHERE api_key_hash = %s
            """,
            (key_hash,),
        ).fetchone()
    if not row:
        return None
    if not row[6]:
        return None
    return Consumer(
        id=int(row[0]),
        tenant_id=row[1],
        name=row[2],
        tier=row[3],
        scopes=list(row[4] or []),
        rate_limit_rpm=int(row[5]),
        is_active=bool(row[6]),
    )


def extract_api_key(request: Request) -> str | None:
    header = request.headers.get("x-api-key")
    if header and header.strip():
        return header.strip()
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return None


def record_usage(
    *,
    consumer: Consumer | None,
    endpoint: str,
    method: str,
    status_code: int,
    latency_ms: int | None = None,
    request_id: str | None = None,
    config: FinanceConfig | None = None,
) -> None:
    cfg = config or load_config()
    try:
        with db.connect(cfg) as conn:
            conn.execute(
                """
                INSERT INTO finance_api_usage
                    (consumer_id, tenant_id, endpoint, method, status_code,
                     latency_ms, request_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    consumer.id if consumer else None,
                    consumer.tenant_id if consumer else "anonymous",
                    endpoint,
                    method,
                    status_code,
                    latency_ms,
                    request_id or uuid.uuid4().hex,
                ),
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("usage_record_failed", extra={"error": str(exc)})


def require_scopes(*needed: str) -> Callable:
    """FastAPI dependency factory: authenticate + check scopes + rate limit."""

    async def _dep(request: Request) -> Consumer | None:
        from projects.finance_agent.rate_limit import check_rate_limit

        cfg = load_config()
        raw = extract_api_key(request)
        consumer: Consumer | None = None

        def _deny(status: int, detail: str, headers: dict | None = None) -> None:
            record_usage(
                consumer=consumer,
                endpoint=request.url.path,
                method=request.method,
                status_code=status,
                request_id=request.headers.get("x-request-id"),
                config=cfg,
            )
            raise HTTPException(status_code=status, detail=detail, headers=headers)

        if raw:
            consumer = lookup_consumer_by_key(raw, config=cfg)
            if consumer is None:
                _deny(401, "Invalid API key")
        elif cfg.api_auth_required:
            _deny(401, "Missing API key. Pass X-API-Key or Authorization: Bearer")
        else:
            return None

        assert consumer is not None
        for scope in needed:
            if not consumer.has_scope(scope):
                _deny(403, f"Missing scope: {scope}")

        allowed, retry_after = check_rate_limit(consumer, config=cfg)
        if not allowed:
            _deny(
                429,
                "Rate limit exceeded",
                headers={"Retry-After": str(retry_after or 60)},
            )

        request.state.finance_consumer = consumer
        return consumer

    return _dep
