"""Postgres helpers for finance agent (F1+)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from projects.finance_agent.config import FinanceConfig, load_config


def require_database_url(config: FinanceConfig | None = None) -> str:
    cfg = config or load_config()
    if not cfg.database_url:
        raise RuntimeError(
            "DATABASE_URL is required. Example: "
            "postgresql://langgraph:langgraph@localhost:5433/langgraph"
        )
    return cfg.database_url


@contextmanager
def connect(config: FinanceConfig | None = None) -> Iterator:
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "Install psycopg: pip install 'psycopg[binary]'"
        ) from exc

    url = require_database_url(config)
    conn = psycopg.connect(url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ping(config: FinanceConfig | None = None) -> bool:
    with connect(config) as conn:
        row = conn.execute("SELECT 1").fetchone()
        return bool(row and row[0] == 1)
