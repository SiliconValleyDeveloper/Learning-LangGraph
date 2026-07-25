"""Fundamentals ingest + query helpers (F5).

Loads bundled sample annual/quarterly statements into finance_fundamentals.
Values are synthetic for learning (INR crore).

CLI:
    python -m projects.finance_agent.fundamentals ingest
    python -m projects.finance_agent.fundamentals show RELIANCE
    python -m projects.finance_agent.fundamentals status
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from projects.finance_agent import db
from projects.finance_agent.config import FinanceConfig, load_config
from projects.finance_agent.logging_util import get_logger

log = get_logger("finance.fundamentals")
DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_FILE = DATA_DIR / "sample_fundamentals.csv"

VALID_PERIOD_TYPES = {"annual", "quarterly"}
VALID_STATEMENTS = {"income_statement", "balance_sheet", "cash_flow"}


@dataclass
class FundamentalRow:
    symbol: str
    exchange: str
    period_type: str
    period: str
    statement: str
    line_item: str
    value: float | None
    currency: str
    unit: str | None


def load_sample_fundamentals() -> tuple[list[FundamentalRow], str]:
    text = SAMPLE_FILE.read_text(encoding="utf-8")
    rows: list[FundamentalRow] = []
    for raw in csv.DictReader(text.splitlines()):
        symbol = (raw.get("symbol") or "").strip().upper()
        period_type = (raw.get("period_type") or "").strip().lower()
        period = (raw.get("period") or "").strip()
        statement = (raw.get("statement") or "").strip().lower()
        line_item = (raw.get("line_item") or "").strip()
        if not symbol or not period or not line_item:
            continue
        if period_type not in VALID_PERIOD_TYPES:
            continue
        if statement not in VALID_STATEMENTS:
            statement = "balance_sheet"
        value_raw = (raw.get("value") or "").strip()
        try:
            value = float(value_raw) if value_raw else None
        except ValueError:
            value = None
        exchange = (raw.get("exchange") or "NSE").strip().upper()
        if exchange not in {"NSE", "BSE", "OTHER"}:
            exchange = "NSE"
        rows.append(
            FundamentalRow(
                symbol=symbol,
                exchange=exchange,
                period_type=period_type,
                period=period,
                statement=statement,
                line_item=line_item,
                value=value,
                currency=(raw.get("currency") or "INR").strip() or "INR",
                unit=(raw.get("unit") or "").strip() or None,
            )
        )
    log.info("fundamentals_loaded", extra={"count": len(rows), "source": "sample"})
    return rows, "sample"


def upsert_fundamentals(
    rows: list[FundamentalRow],
    source: str,
    *,
    config: FinanceConfig | None = None,
) -> int:
    cfg = config or load_config()
    n = 0
    with db.connect(cfg) as conn:
        for r in rows:
            conn.execute(
                """
                INSERT INTO finance_fundamentals
                    (symbol, exchange, period_type, period, statement,
                     line_item, value, currency, unit, source, fetched_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (exchange, symbol, period_type, period, statement, line_item, source)
                DO UPDATE SET
                    value = EXCLUDED.value,
                    currency = EXCLUDED.currency,
                    unit = EXCLUDED.unit,
                    fetched_at = EXCLUDED.fetched_at
                """,
                (
                    r.symbol,
                    r.exchange,
                    r.period_type,
                    r.period,
                    r.statement,
                    r.line_item,
                    r.value,
                    r.currency,
                    r.unit,
                    source,
                ),
            )
            n += 1
    return n


def ingest_sample(*, config: FinanceConfig | None = None) -> dict[str, Any]:
    cfg = config or load_config()
    rows, source = load_sample_fundamentals()
    count = upsert_fundamentals(rows, source, config=cfg)
    log.info("fundamentals_ingested", extra={"rows": count, "source": source})
    return {"source": source, "rows_ok": count}


def query_fundamentals(
    symbol: str,
    *,
    exchange: str | None = None,
    period_type: str | None = None,
    period: str | None = None,
    statement: str | None = None,
    config: FinanceConfig | None = None,
) -> dict[str, Any]:
    cfg = config or load_config()
    clauses = ["symbol = %s"]
    params: list[Any] = [symbol.upper()]
    if exchange:
        clauses.append("exchange = %s")
        params.append(exchange.upper())
    if period_type:
        clauses.append("period_type = %s")
        params.append(period_type.lower())
    if period:
        clauses.append("period = %s")
        params.append(period)
    if statement:
        clauses.append("statement = %s")
        params.append(statement.lower())
    where = " AND ".join(clauses)

    period_clauses = ["symbol = %s"]
    period_params: list[Any] = [symbol.upper()]
    if exchange:
        period_clauses.append("exchange = %s")
        period_params.append(exchange.upper())
    period_where = " AND ".join(period_clauses)

    with db.connect(cfg) as conn:
        rows = conn.execute(
            f"""
            SELECT symbol, exchange, period_type, period, statement,
                   line_item, value, currency, unit, source, fetched_at
            FROM finance_fundamentals
            WHERE {where}
            ORDER BY
                CASE period_type WHEN 'annual' THEN 0 ELSE 1 END,
                period DESC,
                statement,
                line_item
            """,
            params,
        ).fetchall()

        # Always list all periods for the symbol (ignore period/statement filters)
        periods = conn.execute(
            f"""
            SELECT period_type, period
            FROM (
                SELECT DISTINCT period_type, period
                FROM finance_fundamentals
                WHERE {period_where}
            ) p
            ORDER BY
                CASE period_type WHEN 'annual' THEN 0 ELSE 1 END,
                period DESC
            """,
            period_params,
        ).fetchall()

    lines = [
        {
            "symbol": r[0],
            "exchange": r[1],
            "period_type": r[2],
            "period": r[3],
            "statement": r[4],
            "line_item": r[5],
            "value": float(r[6]) if r[6] is not None else None,
            "currency": r[7],
            "unit": r[8],
            "source": r[9],
            "fetched_at": str(r[10]),
        }
        for r in rows
    ]

    # Group for UI convenience
    by_period: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for line in lines:
        key = f"{line['period_type']}:{line['period']}"
        by_period.setdefault(key, {})
        by_period[key].setdefault(line["statement"], []).append(line)

    return {
        "symbol": symbol.upper(),
        "count": len(lines),
        "available_periods": [
            {"period_type": p[0], "period": p[1]} for p in periods
        ],
        "lines": lines,
        "by_period": by_period,
    }


def status(*, config: FinanceConfig | None = None) -> dict[str, Any]:
    cfg = config or load_config()
    with db.connect(cfg) as conn:
        total = conn.execute("SELECT count(*) FROM finance_fundamentals").fetchone()[0]
        by_sym = conn.execute(
            """
            SELECT symbol, count(*)
            FROM finance_fundamentals
            GROUP BY symbol
            ORDER BY symbol
            """
        ).fetchall()
        periods = conn.execute(
            """
            SELECT period_type, count(DISTINCT period)
            FROM finance_fundamentals
            GROUP BY period_type
            """
        ).fetchall()
    return {
        "rows": int(total),
        "by_symbol": {s: int(c) for s, c in by_sym},
        "period_types": {p: int(c) for p, c in periods},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finance F5 fundamentals")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="Load sample fundamentals into Postgres")
    sub.add_parser("status", help="Row counts")
    p_show = sub.add_parser("show", help="Show fundamentals for a symbol")
    p_show.add_argument("symbol")
    p_show.add_argument("--period-type", choices=sorted(VALID_PERIOD_TYPES))
    p_show.add_argument("--period")
    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        print(json.dumps(ingest_sample(), indent=2))
        print(json.dumps({"status": status()}, indent=2))
        return 0
    if args.cmd == "status":
        print(json.dumps(status(), indent=2))
        return 0
    if args.cmd == "show":
        print(
            json.dumps(
                query_fundamentals(
                    args.symbol,
                    period_type=args.period_type,
                    period=args.period,
                ),
                indent=2,
            )
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
