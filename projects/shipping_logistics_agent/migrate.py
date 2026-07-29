"""Apply the isolated shipping schema and sample data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from projects.shipping_logistics_agent import db

SQL_DIR = Path(__file__).resolve().parent / "sql"


def apply_all() -> dict[str, object]:
    files = sorted(SQL_DIR.glob("[0-9]*.sql"))
    if not files:
        raise FileNotFoundError(f"No SQL migrations found in {SQL_DIR}")

    with db.connect() as conn:
        conn.execute("CREATE SCHEMA IF NOT EXISTS shipping")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS shipping.schema_migrations (
                version TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        applied = {
            row[0]
            for row in conn.execute(
                "SELECT version FROM shipping.schema_migrations"
            ).fetchall()
        }

    newly: list[str] = []
    for path in files:
        if path.stem in applied:
            continue
        with db.connect() as conn:
            conn.execute(path.read_text(encoding="utf-8"))
        newly.append(path.stem)
    return {"newly_applied": newly, **status()}


def status() -> dict[str, object]:
    with db.connect() as conn:
        versions = [
            row[0]
            for row in conn.execute(
                """
                SELECT version
                FROM shipping.schema_migrations
                ORDER BY applied_at
                """
            ).fetchall()
        ]
        tables = [
            row[0]
            for row in conn.execute(
                """
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'shipping'
                ORDER BY tablename
                """
            ).fetchall()
        ]
        counts = {
            "customers": conn.execute(
                "SELECT count(*) FROM shipping.customers"
            ).fetchone()[0],
            "ports": conn.execute(
                "SELECT count(*) FROM shipping.ports"
            ).fetchone()[0],
            "vessels": conn.execute(
                "SELECT count(*) FROM shipping.vessels"
            ).fetchone()[0],
            "sailings": conn.execute(
                "SELECT count(*) FROM shipping.sailings"
            ).fetchone()[0],
        }
    return {
        "postgres_ok": True,
        "migrations": versions,
        "tables": tables,
        "sample_counts": {key: int(value) for key, value in counts.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shipping schema migrations")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    result = status() if args.status else apply_all()
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

