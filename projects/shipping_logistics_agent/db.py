"""PostgreSQL helpers restricted to the isolated shipping schema."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from projects.shipping_logistics_agent.config import ShippingConfig, load_config


@contextmanager
def connect(config: ShippingConfig | None = None) -> Iterator:
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "Install psycopg: pip install 'psycopg[binary]'"
        ) from exc

    cfg = config or load_config()
    conn = psycopg.connect(cfg.database_url)
    try:
        from psycopg import sql

        with conn.cursor() as cursor:
            cursor.execute(
                sql.SQL("SET LOCAL search_path TO {}, public").format(
                    sql.Identifier(cfg.schema)
                )
            )
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ping(config: ShippingConfig | None = None) -> bool:
    with connect(config) as conn:
        return conn.execute("SELECT 1").fetchone()[0] == 1

