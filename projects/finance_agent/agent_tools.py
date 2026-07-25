"""Read-only finance tools used by the F7 LangGraph agent."""

from __future__ import annotations

from typing import Any

from projects.finance_agent import db, filings, fundamentals, quotes
from projects.finance_agent.config import FinanceConfig, load_config


def quote_tool(
    symbol: str,
    exchange: str = "NSE",
    *,
    config: FinanceConfig | None = None,
) -> dict[str, Any]:
    cfg = config or load_config()
    quote = quotes.get_quote(
        symbol,
        exchange=exchange,
        refresh_if_missing=True,
        config=cfg,
    )
    return quote.to_dict() if quote else {}


def fundamentals_tool(
    symbol: str,
    exchange: str = "NSE",
    *,
    config: FinanceConfig | None = None,
) -> dict[str, Any]:
    return fundamentals.query_fundamentals(
        symbol,
        exchange=exchange,
        config=config,
    )


def corp_actions_tool(
    symbol: str,
    *,
    limit: int = 10,
    config: FinanceConfig | None = None,
) -> list[dict[str, Any]]:
    cfg = config or load_config()
    with db.connect(cfg) as conn:
        rows = conn.execute(
            """
            SELECT action_type, ex_date, record_date, ratio, amount,
                   currency, source
            FROM finance_corp_actions
            WHERE symbol = %s
            ORDER BY ex_date DESC NULLS LAST, id DESC
            LIMIT %s
            """,
            (symbol.upper(), limit),
        ).fetchall()
    return [
        {
            "action_type": row[0],
            "ex_date": str(row[1]) if row[1] else None,
            "record_date": str(row[2]) if row[2] else None,
            "ratio": row[3],
            "amount": float(row[4]) if row[4] is not None else None,
            "currency": row[5],
            "source": row[6],
        }
        for row in rows
    ]


def filings_tool(
    symbol: str,
    question: str,
    *,
    config: FinanceConfig | None = None,
) -> dict[str, Any]:
    return filings.search_filings(question, symbol=symbol, config=config)
