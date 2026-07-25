"""HTTP routes for finance quotes (F3/F3.5) — analysis API with consumer auth."""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from projects.finance_agent import auth
from projects.finance_agent import ingest as finance_ingest
from projects.finance_agent import kite_client, quotes, redis_client
from projects.finance_agent.auth import Consumer
from projects.finance_agent.config import load_config

router = APIRouter(prefix="/api/finance", tags=["finance"])


class RefreshRequest(BaseModel):
    symbols: list[str] | None = None
    exchange: str = Field(default="NSE", max_length=8)


class AgentRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    symbol: str | None = Field(default=None, max_length=32)
    exchange: str = Field(default="NSE", max_length=8)
    level: str = Field(default="L2", pattern="^(L1|L2|l1|l2)$")


def _audit(
    request: Request,
    *,
    consumer: Consumer | None,
    status_code: int,
    started: float,
) -> None:
    auth.record_usage(
        consumer=consumer,
        endpoint=request.url.path,
        method=request.method,
        status_code=status_code,
        latency_ms=int((time.perf_counter() - started) * 1000),
        request_id=request.headers.get("x-request-id") or uuid.uuid4().hex,
    )


@router.get("/status")
def finance_status() -> dict[str, Any]:
    """Public health / capability probe (no API key required)."""
    cfg = load_config()
    quote_status = quotes.status(cfg)
    try:
        ingest_status = finance_ingest.status(cfg)
    except Exception as exc:  # noqa: BLE001
        ingest_status = {"error": str(exc)}
    try:
        from projects.finance_agent import filings as fin

        filings_status = fin.status(config=cfg)
    except Exception as exc:  # noqa: BLE001
        filings_status = {"error": str(exc)}
    try:
        consumers = len(auth.list_consumers(config=cfg))
    except Exception as exc:  # noqa: BLE001
        consumers = None
        consumers_error = str(exc)
    else:
        consumers_error = None
    return {
        "enabled": cfg.enabled,
        "product": "markets-analysis",
        "broker": False,
        "orders_supported": False,
        "api_auth_required": cfg.api_auth_required,
        "auth": {
            "header": "X-API-Key or Authorization: Bearer",
            "active_consumers": consumers,
            "error": consumers_error,
            "create_cli": "python -m projects.finance_agent.consumers create --name demo --tier free",
        },
        "quote": quote_status,
        "ingest": ingest_status,
        "filings": filings_status,
        "agent": {
            "phase": "F7",
            "levels": ["L1", "L2"],
            "analysis_only": True,
            "orders_supported": False,
            "endpoint": "/api/finance/agent/analyse",
            "required_scope": "research:read",
        },
        "redis": redis_client.health(cfg),
        "kite_credentials_ready": kite_client.kite_credentials_ready(cfg),
    }


@router.get("/quote/{symbol}")
async def get_quote(
    request: Request,
    symbol: str,
    exchange: str = Query(default="NSE", max_length=8),
    refresh: bool = Query(default=True),
    consumer: Consumer | None = Depends(auth.require_scopes("quotes:read")),
) -> dict[str, Any]:
    started = time.perf_counter()

    def _get() -> dict[str, Any]:
        q = quotes.get_quote(
            symbol,
            exchange=exchange,
            refresh_if_missing=refresh,
        )
        if q is None:
            raise HTTPException(status_code=404, detail=f"Quote not found for {symbol}")
        return q.to_dict()

    try:
        data = await run_in_threadpool(_get)
        _audit(request, consumer=consumer, status_code=200, started=started)
        return data
    except HTTPException as exc:
        _audit(request, consumer=consumer, status_code=exc.status_code, started=started)
        raise
    except Exception as exc:  # noqa: BLE001
        _audit(request, consumer=consumer, status_code=500, started=started)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/quotes")
async def list_quotes(
    request: Request,
    symbols: str | None = Query(
        default=None,
        description="Comma-separated symbols; default FINANCE_QUOTE_SYMBOLS",
    ),
    exchange: str = Query(default="NSE", max_length=8),
    refresh: bool = Query(default=False, description="Force refresh before list"),
    consumer: Consumer | None = Depends(auth.require_scopes("quotes:read")),
) -> dict[str, Any]:
    started = time.perf_counter()
    sym_list = (
        [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if symbols
        else None
    )

    def _list() -> dict[str, Any]:
        if refresh or not quotes.list_cached_quotes(sym_list, exchange=exchange):
            result = quotes.refresh_quotes(sym_list, exchange=exchange)
            return {
                "source": result["source"],
                "count": result["count"],
                "quotes": result["quotes"],
            }
        cached = quotes.list_cached_quotes(sym_list, exchange=exchange)
        return {
            "source": "cache",
            "count": len(cached),
            "quotes": [q.to_dict() for q in cached],
        }

    try:
        data = await run_in_threadpool(_list)
        _audit(request, consumer=consumer, status_code=200, started=started)
        return data
    except Exception as exc:  # noqa: BLE001
        _audit(request, consumer=consumer, status_code=500, started=started)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/quotes/refresh")
async def refresh_quotes(
    request: Request,
    body: RefreshRequest | None = None,
    consumer: Consumer | None = Depends(auth.require_scopes("quotes:refresh")),
) -> dict[str, Any]:
    started = time.perf_counter()
    req = body or RefreshRequest()

    def _refresh() -> dict[str, Any]:
        return quotes.refresh_quotes(req.symbols, exchange=req.exchange)

    try:
        data = await run_in_threadpool(_refresh)
        _audit(request, consumer=consumer, status_code=200, started=started)
        return data
    except Exception as exc:  # noqa: BLE001
        _audit(request, consumer=consumer, status_code=500, started=started)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/companies")
async def list_companies(
    request: Request,
    exchange: str | None = Query(default=None, max_length=8),
    limit: int = Query(default=100, ge=1, le=500),
    consumer: Consumer | None = Depends(auth.require_scopes("quotes:read")),
) -> dict[str, Any]:
    started = time.perf_counter()

    def _list() -> dict[str, Any]:
        from projects.finance_agent import db

        cfg = load_config()
        clauses = ["1=1"]
        params: list[Any] = []
        if exchange:
            clauses.append("exchange = %s")
            params.append(exchange.upper())
        params.append(limit)
        where = " AND ".join(clauses)
        with db.connect(cfg) as conn:
            rows = conn.execute(
                f"""
                SELECT symbol, exchange, isin, name, series, status, sector, industry
                FROM finance_company_master
                WHERE {where}
                ORDER BY exchange, symbol
                LIMIT %s
                """,
                params,
            ).fetchall()
        return {
            "count": len(rows),
            "companies": [
                {
                    "symbol": r[0],
                    "exchange": r[1],
                    "isin": r[2],
                    "name": r[3],
                    "series": r[4],
                    "status": r[5],
                    "sector": r[6],
                    "industry": r[7],
                }
                for r in rows
            ],
        }

    try:
        data = await run_in_threadpool(_list)
        _audit(request, consumer=consumer, status_code=200, started=started)
        return data
    except Exception as exc:  # noqa: BLE001
        _audit(request, consumer=consumer, status_code=500, started=started)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/corp-actions")
async def list_corp_actions(
    request: Request,
    symbol: str | None = Query(default=None, max_length=32),
    limit: int = Query(default=50, ge=1, le=200),
    consumer: Consumer | None = Depends(auth.require_scopes("quotes:read")),
) -> dict[str, Any]:
    started = time.perf_counter()

    def _list() -> dict[str, Any]:
        from projects.finance_agent import db

        cfg = load_config()
        clauses = ["1=1"]
        params: list[Any] = []
        if symbol:
            clauses.append("symbol = %s")
            params.append(symbol.upper())
        params.append(limit)
        where = " AND ".join(clauses)
        with db.connect(cfg) as conn:
            rows = conn.execute(
                f"""
                SELECT symbol, exchange, action_type, ex_date, record_date,
                       ratio, amount, currency, source
                FROM finance_corp_actions
                WHERE {where}
                ORDER BY ex_date DESC NULLS LAST, id DESC
                LIMIT %s
                """,
                params,
            ).fetchall()
        return {
            "count": len(rows),
            "actions": [
                {
                    "symbol": r[0],
                    "exchange": r[1],
                    "action_type": r[2],
                    "ex_date": str(r[3]) if r[3] else None,
                    "record_date": str(r[4]) if r[4] else None,
                    "ratio": r[5],
                    "amount": float(r[6]) if r[6] is not None else None,
                    "currency": r[7],
                    "source": r[8],
                }
                for r in rows
            ],
        }

    try:
        data = await run_in_threadpool(_list)
        _audit(request, consumer=consumer, status_code=200, started=started)
        return data
    except Exception as exc:  # noqa: BLE001
        _audit(request, consumer=consumer, status_code=500, started=started)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/fundamentals/{symbol}")
async def get_fundamentals(
    request: Request,
    symbol: str,
    exchange: str | None = Query(default=None, max_length=8),
    period_type: str | None = Query(default=None, pattern="^(annual|quarterly)$"),
    period: str | None = Query(default=None, max_length=32),
    statement: str | None = Query(
        default=None, pattern="^(income_statement|balance_sheet|cash_flow)$"
    ),
    consumer: Consumer | None = Depends(auth.require_scopes("quotes:read")),
) -> dict[str, Any]:
    """Annual / quarterly statements for a symbol (F5)."""
    started = time.perf_counter()

    def _get() -> dict[str, Any]:
        from projects.finance_agent import fundamentals as fund

        return fund.query_fundamentals(
            symbol,
            exchange=exchange,
            period_type=period_type,
            period=period,
            statement=statement,
        )

    try:
        data = await run_in_threadpool(_get)
        if data.get("count", 0) == 0:
            _audit(request, consumer=consumer, status_code=404, started=started)
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No fundamentals for {symbol.upper()}. "
                    "Run: python -m projects.finance_agent.fundamentals ingest"
                ),
            )
        _audit(request, consumer=consumer, status_code=200, started=started)
        return data
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        _audit(request, consumer=consumer, status_code=500, started=started)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/filings")
async def list_filings(
    request: Request,
    symbol: str | None = Query(default=None, max_length=32),
    consumer: Consumer | None = Depends(auth.require_scopes("quotes:read")),
) -> dict[str, Any]:
    """List ingested filing documents (F6)."""
    started = time.perf_counter()

    def _list() -> dict[str, Any]:
        from projects.finance_agent import filings as fin

        return fin.list_documents(symbol)

    try:
        data = await run_in_threadpool(_list)
        _audit(request, consumer=consumer, status_code=200, started=started)
        return data
    except Exception as exc:  # noqa: BLE001
        _audit(request, consumer=consumer, status_code=500, started=started)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/filings/{symbol}/search")
async def search_filings(
    request: Request,
    symbol: str,
    q: str = Query(..., min_length=2, max_length=500, description="Research question"),
    consumer: Consumer | None = Depends(auth.require_scopes("quotes:read")),
) -> dict[str, Any]:
    """pgvector retrieve → dynamic CrossEncoder/lexical rerank (F6)."""
    started = time.perf_counter()

    def _search() -> dict[str, Any]:
        from projects.finance_agent import filings as fin

        return fin.search_filings(q, symbol=symbol)

    try:
        data = await run_in_threadpool(_search)
        _audit(request, consumer=consumer, status_code=200, started=started)
        return data
    except ValueError as exc:
        _audit(request, consumer=consumer, status_code=400, started=started)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        _audit(request, consumer=consumer, status_code=500, started=started)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/agent/analyse")
async def analyse_with_agent(
    request: Request,
    body: AgentRequest,
    consumer: Consumer | None = Depends(auth.require_scopes("research:read")),
) -> dict[str, Any]:
    """Run the F7 read-only LangGraph L1/L2 analysis workflow."""
    started = time.perf_counter()

    def _analyse() -> dict[str, Any]:
        from projects.finance_agent.agent import analyse

        return analyse(
            body.question,
            symbol=body.symbol,
            exchange=body.exchange,
            level=body.level,
        )

    try:
        data = await run_in_threadpool(_analyse)
        _audit(request, consumer=consumer, status_code=200, started=started)
        return data
    except ValueError as exc:
        _audit(request, consumer=consumer, status_code=400, started=started)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        _audit(request, consumer=consumer, status_code=500, started=started)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
