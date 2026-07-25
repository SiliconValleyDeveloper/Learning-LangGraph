"""Quote service (F3): Zerodha/sample → Redis TTL → optional finance_ticks.

Analysis only — no order placement.

CLI:
    python -m projects.finance_agent.quotes refresh
    python -m projects.finance_agent.quotes get RELIANCE
    python -m projects.finance_agent.quotes status
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from projects.finance_agent import db, kite_client, redis_client
from projects.finance_agent.config import FinanceConfig, load_config
from projects.finance_agent.kite_client import Quote
from projects.finance_agent.logging_util import get_logger

log = get_logger("finance.quotes")


def quote_key(symbol: str, exchange: str = "NSE") -> str:
    return f"quote:{exchange.upper()}:{symbol.upper()}"


def cache_quote(quote: Quote, *, config: FinanceConfig | None = None) -> None:
    cfg = config or load_config()
    key = quote_key(quote.symbol, quote.exchange)
    redis_client.set_json(
        key,
        json.dumps(quote.to_dict()),
        ttl_seconds=cfg.quote_ttl_seconds,
        config=cfg,
    )


def publish_quote(quote: Quote, *, config: FinanceConfig | None = None) -> None:
    cfg = config or load_config()
    try:
        redis_client.publish(
            cfg.quote_stream_channel,
            json.dumps(quote.to_dict()),
            config=cfg,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("quote_publish_failed", extra={"error": str(exc)})


def get_cached_quote(
    symbol: str,
    *,
    exchange: str = "NSE",
    config: FinanceConfig | None = None,
) -> Quote | None:
    cfg = config or load_config()
    raw = redis_client.get_json(quote_key(symbol, exchange), config=cfg)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return Quote(**{k: data.get(k) for k in Quote.__dataclass_fields__})
    except Exception as exc:  # noqa: BLE001
        log.warning("cache_parse_failed", extra={"symbol": symbol, "error": str(exc)})
        return None


def persist_tick(quote: Quote, *, config: FinanceConfig | None = None) -> None:
    cfg = config or load_config()
    if not cfg.quote_persist_ticks:
        return
    with db.connect(cfg) as conn:
        conn.execute(
            """
            INSERT INTO finance_ticks
                (symbol, exchange, ts, ltp, volume, oi, bid, ask, source, fetched_at)
            VALUES (%s, %s, COALESCE(%s::timestamptz, now()), %s, %s, %s, %s, %s, %s, now())
            """,
            (
                quote.symbol,
                quote.exchange,
                quote.ts,
                quote.ltp,
                quote.volume,
                quote.oi,
                quote.bid,
                quote.ask,
                quote.source,
            ),
        )


def refresh_quotes(
    symbols: list[str] | None = None,
    *,
    config: FinanceConfig | None = None,
    exchange: str = "NSE",
) -> dict[str, Any]:
    """Pull from Kite/sample, write Redis + ticks, publish stream."""
    cfg = config or load_config()
    quotes, source = kite_client.fetch_quotes(symbols, config=cfg, exchange=exchange)
    for q in quotes:
        cache_quote(q, config=cfg)
        publish_quote(q, config=cfg)
        try:
            persist_tick(q, config=cfg)
        except Exception as exc:  # noqa: BLE001
            log.warning("tick_persist_failed", extra={"symbol": q.symbol, "error": str(exc)})

    log.info(
        "quotes_refreshed",
        extra={"count": len(quotes), "source": source, "ttl": cfg.quote_ttl_seconds},
    )
    return {
        "source": source,
        "count": len(quotes),
        "ttl_seconds": cfg.quote_ttl_seconds,
        "quotes": [q.to_dict() for q in quotes],
    }


def get_quote(
    symbol: str,
    *,
    exchange: str = "NSE",
    refresh_if_missing: bool = True,
    config: FinanceConfig | None = None,
) -> Quote | None:
    """Redis-first quote; optional refresh on miss."""
    cfg = config or load_config()
    cached = get_cached_quote(symbol, exchange=exchange, config=cfg)
    if cached is not None:
        return cached
    if not refresh_if_missing:
        return None
    result = refresh_quotes([symbol], config=cfg, exchange=exchange)
    for q in result.get("quotes") or []:
        if q.get("symbol", "").upper() == symbol.upper():
            return Quote(**{k: q.get(k) for k in Quote.__dataclass_fields__})
    return None


def list_cached_quotes(
    symbols: list[str] | None = None,
    *,
    exchange: str = "NSE",
    config: FinanceConfig | None = None,
) -> list[Quote]:
    cfg = config or load_config()
    syms = [s.upper() for s in (symbols or list(cfg.quote_default_symbols))]
    out: list[Quote] = []
    for s in syms:
        q = get_cached_quote(s, exchange=exchange, config=cfg)
        if q:
            out.append(q)
    return out


def status(config: FinanceConfig | None = None) -> dict[str, Any]:
    cfg = config or load_config()
    mode = kite_client.resolve_quote_source(cfg)
    ticks = None
    try:
        with db.connect(cfg) as conn:
            ticks = conn.execute("SELECT count(*) FROM finance_ticks").fetchone()[0]
            latest = conn.execute(
                """
                SELECT symbol, exchange, ltp, source, ts
                FROM finance_ticks
                ORDER BY ts DESC
                LIMIT 5
                """
            ).fetchall()
    except Exception as exc:  # noqa: BLE001
        latest = []
        tick_error = str(exc)
    else:
        tick_error = None

    cached = []
    for s in cfg.quote_default_symbols[:5]:
        q = get_cached_quote(s, config=cfg)
        if q:
            cached.append({"symbol": q.symbol, "ltp": q.ltp, "source": q.source})

    return {
        "quote_source_mode": cfg.quote_source,
        "resolved_source": mode,
        "kite_credentials_ready": kite_client.kite_credentials_ready(cfg),
        "ttl_seconds": cfg.quote_ttl_seconds,
        "persist_ticks": cfg.quote_persist_ticks,
        "redis": redis_client.health(cfg),
        "ticks_total": int(ticks) if ticks is not None else None,
        "ticks_error": tick_error,
        "latest_ticks": [
            {
                "symbol": r[0],
                "exchange": r[1],
                "ltp": float(r[2]) if r[2] is not None else None,
                "source": r[3],
                "ts": str(r[4]),
            }
            for r in latest
        ],
        "cached_sample": cached,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finance F3 quotes (Kite/sample → Redis)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_refresh = sub.add_parser("refresh", help="Fetch quotes → Redis + ticks")
    p_refresh.add_argument(
        "symbols",
        nargs="*",
        help="Symbols (default: FINANCE_QUOTE_SYMBOLS)",
    )
    p_refresh.add_argument("--exchange", default="NSE")

    p_get = sub.add_parser("get", help="Get one quote (cache-first)")
    p_get.add_argument("symbol")
    p_get.add_argument("--exchange", default="NSE")
    p_get.add_argument("--no-refresh", action="store_true")

    sub.add_parser("status", help="Quote pipeline status")

    args = parser.parse_args(argv)

    if args.cmd == "status":
        print(json.dumps(status(), indent=2, default=str))
        return 0

    if args.cmd == "refresh":
        result = refresh_quotes(
            args.symbols or None,
            exchange=args.exchange,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.cmd == "get":
        q = get_quote(
            args.symbol,
            exchange=args.exchange,
            refresh_if_missing=not args.no_refresh,
        )
        if q is None:
            print(json.dumps({"error": "not found", "symbol": args.symbol}))
            return 1
        print(json.dumps(q.to_dict(), indent=2, default=str))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
