"""Apply versioned finance SQL migrations.

Usage (from repo root):
    python -m projects.finance_agent.migrate
    python -m projects.finance_agent.migrate --status
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from projects.finance_agent import db
from projects.finance_agent import redis_client
from projects.finance_agent.config import load_config

SQL_DIR = Path(__file__).resolve().parent / "sql"
DOCKER_DB_CONTAINER = "langgraph-pgvector"


def migration_files() -> list[Path]:
    """All versioned SQL migrations in lexical (numeric-prefix) order."""
    return sorted(SQL_DIR.glob("[0-9]*.sql"))


def applied_versions() -> list[str]:
    with db.connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS finance_schema_migrations (
                version TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        rows = conn.execute(
            "SELECT version FROM finance_schema_migrations ORDER BY applied_at"
        ).fetchall()
    return [r[0] for r in rows]


def _run_sql_file(path: Path) -> None:
    """Run a multi-statement SQL file via psql or docker exec."""
    url = db.require_database_url()
    parsed = urlparse(url)
    sql_bytes = path.read_bytes()

    # Prefer docker exec into the lab Postgres container (no local psql needed).
    docker = shutil.which("docker")
    if docker:
        probe = subprocess.run(
            [docker, "inspect", "-f", "{{.State.Running}}", DOCKER_DB_CONTAINER],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0 and probe.stdout.strip() == "true":
            result = subprocess.run(
                [
                    docker,
                    "exec",
                    "-i",
                    DOCKER_DB_CONTAINER,
                    "psql",
                    "-U",
                    "langgraph",
                    "-d",
                    "langgraph",
                    "-v",
                    "ON_ERROR_STOP=1",
                ],
                input=sql_bytes,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "docker exec psql failed:\n"
                    + (result.stderr.decode("utf-8", errors="replace") or result.stdout.decode("utf-8", errors="replace"))
                )
            return

    psql = shutil.which("psql")
    if not psql:
        raise RuntimeError(
            "Cannot apply migration: start Postgres "
            f"(`docker compose -f deploy/docker-compose.yml up -d db`) "
            "or install psql on PATH."
        )

    env = os.environ.copy()
    if parsed.password:
        env["PGPASSWORD"] = parsed.password
    result = subprocess.run(
        [
            psql,
            "-h",
            parsed.hostname or "localhost",
            "-p",
            str(parsed.port or 5432),
            "-U",
            parsed.username or "langgraph",
            "-d",
            (parsed.path or "/langgraph").lstrip("/") or "langgraph",
            "-v",
            "ON_ERROR_STOP=1",
            "-f",
            str(path),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"psql failed:\n{result.stderr or result.stdout}")


def apply_all() -> dict[str, object]:
    """Apply every pending sql/*.sql migration in order (idempotent)."""
    files = migration_files()
    if not files:
        raise FileNotFoundError(f"No migrations found in {SQL_DIR}")

    applied = set(applied_versions())
    newly: list[str] = []
    for path in files:
        version = path.stem
        if version in applied:
            continue
        _run_sql_file(path)
        newly.append(version)

    versions = applied_versions()
    for path in files:
        if path.stem not in versions:
            raise RuntimeError(
                f"Migration {path.stem} ran but was not recorded "
                "in finance_schema_migrations (missing INSERT?)"
            )
    return {
        "applied": bool(newly),
        "newly_applied": newly,
        "already_present": not newly,
        "versions": versions,
    }


# Backwards-compatible alias.
def apply_f1() -> dict[str, object]:
    return apply_all()


def list_finance_tables() -> list[str]:
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public' AND tablename LIKE 'finance_%'
            ORDER BY tablename
            """
        ).fetchall()
    return [r[0] for r in rows]


def status() -> dict[str, object]:
    cfg = load_config()
    out: dict[str, object] = {
        "database_url_set": bool(cfg.database_url),
        "redis_url": cfg.redis_url,
        "finance_enabled": cfg.enabled,
    }
    try:
        out["postgres_ok"] = db.ping(cfg)
        out["migrations"] = applied_versions()
        out["finance_tables"] = list_finance_tables()
    except Exception as exc:  # noqa: BLE001
        out["postgres_ok"] = False
        out["postgres_error"] = str(exc)

    out["redis"] = redis_client.health(cfg)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finance schema migrations")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show Postgres/Redis/migration status without applying",
    )
    args = parser.parse_args(argv)

    if args.status:
        info = status()
        print(info)
        redis_ok = bool((info.get("redis") or {}).get("ok"))
        return 0 if info.get("postgres_ok") and redis_ok else 1

    result = apply_all()
    print(result)
    info = status()
    print("status:", info)
    redis_ok = bool((info.get("redis") or {}).get("ok"))
    tables = info.get("finance_tables") or []
    expected = {path.stem for path in migration_files()}
    if not expected.issubset(set(info.get("migrations") or [])):
        print(f"WARNING: missing migrations {expected}", file=sys.stderr)
        return 1
    if len(tables) < 10:
        print(f"WARNING: expected finance_* tables, got {tables}", file=sys.stderr)
        return 1
    if not redis_ok:
        print(
            "WARNING: Redis not reachable — start with:\n"
            "  docker compose -f deploy/docker-compose.yml up -d redis",
            file=sys.stderr,
        )
        return 2
    print(f"OK — {len(tables)} finance tables, {len(expected)} migrations, Redis up")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
