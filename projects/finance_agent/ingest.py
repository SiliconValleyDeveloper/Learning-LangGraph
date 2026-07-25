"""Daily NSE/BSE knowledge ingest (F2).

Flow:
  Redis lock -> open finance_ingest_runs -> load sources ->
  idempotent upserts (company master, corp actions) -> close run -> release lock.

Run (from repo root):
    python -m projects.finance_agent.ingest
    python -m projects.finance_agent.ingest --status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from typing import Any

from projects.finance_agent import db, redis_client, sources
from projects.finance_agent.config import FinanceConfig, load_config
from projects.finance_agent.logging_util import get_logger

log = get_logger("finance.ingest")

LOCK_NAME = "finance:ingest:daily"
JOB_NAME = "nse_bse_daily"


def _corp_action_dedupe_key(row: sources.CorpActionRow, source: str) -> str:
    raw = "|".join(
        [
            row.exchange or "",
            row.symbol or "",
            row.action_type or "",
            row.ex_date.isoformat() if row.ex_date else "",
            source or "",
        ]
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _start_run(conn, source: str) -> int:
    row = conn.execute(
        """
        INSERT INTO finance_ingest_runs (source, job_name, status, metadata)
        VALUES (%s, %s, 'running', %s)
        RETURNING id
        """,
        (source, JOB_NAME, json.dumps({"mode": source})),
    ).fetchone()
    return int(row[0])


def _finish_run(
    conn,
    run_id: int,
    *,
    status: str,
    rows_ok: int,
    rows_failed: int,
    file_names: list[str],
    error: str | None,
    metadata: dict[str, Any],
) -> None:
    conn.execute(
        """
        UPDATE finance_ingest_runs
        SET status = %s,
            rows_ok = %s,
            rows_failed = %s,
            file_names = %s,
            error_summary = %s,
            finished_at = now(),
            metadata = %s
        WHERE id = %s
        """,
        (
            status,
            rows_ok,
            rows_failed,
            file_names,
            error,
            json.dumps(metadata),
            run_id,
        ),
    )


def _upsert_companies(conn, rows: list[sources.CompanyRow], source: str) -> int:
    n = 0
    for r in rows:
        conn.execute(
            """
            INSERT INTO finance_company_master
                (symbol, exchange, isin, name, series, status,
                 sector, industry, source, fetched_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (exchange, symbol) DO UPDATE SET
                isin = EXCLUDED.isin,
                name = EXCLUDED.name,
                series = EXCLUDED.series,
                status = EXCLUDED.status,
                sector = EXCLUDED.sector,
                industry = EXCLUDED.industry,
                source = EXCLUDED.source,
                fetched_at = EXCLUDED.fetched_at,
                updated_at = now()
            """,
            (
                r.symbol, r.exchange, r.isin, r.name, r.series,
                r.status, r.sector, r.industry, source,
            ),
        )
        n += 1
    return n


def _upsert_corp_actions(conn, rows: list[sources.CorpActionRow], source: str) -> int:
    n = 0
    for r in rows:
        key = _corp_action_dedupe_key(r, source)
        conn.execute(
            """
            INSERT INTO finance_corp_actions
                (symbol, exchange, action_type, ex_date, record_date,
                 ratio, amount, currency, details, source, fetched_at, dedupe_key)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'INR', %s, %s, now(), %s)
            ON CONFLICT (dedupe_key) DO UPDATE SET
                ex_date = EXCLUDED.ex_date,
                record_date = EXCLUDED.record_date,
                ratio = EXCLUDED.ratio,
                amount = EXCLUDED.amount,
                details = EXCLUDED.details,
                source = EXCLUDED.source,
                fetched_at = EXCLUDED.fetched_at
            """,
            (
                r.symbol, r.exchange, r.action_type, r.ex_date, r.record_date,
                r.ratio, r.amount, json.dumps({}), source, key,
            ),
        )
        n += 1
    return n


def run_ingest(config: FinanceConfig | None = None) -> dict[str, Any]:
    cfg = config or load_config()
    if not cfg.nse_bse_ingest_enabled:
        log.warning("ingest_disabled")
        return {"skipped": True, "reason": "NSE_BSE_INGEST_ENABLED is false"}

    token = uuid.uuid4().hex
    have_lock = False
    try:
        have_lock = redis_client.acquire_lock(
            LOCK_NAME, ttl_seconds=cfg.ingest_lock_ttl, token=token, config=cfg
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("redis_lock_unavailable", extra={"error": str(exc)})
        if cfg.ingest_require_lock:
            raise
    else:
        if not have_lock:
            log.warning("ingest_already_running")
            return {"skipped": True, "reason": "another ingest holds the lock"}

    log.info("ingest_start", extra={"lock": have_lock, "mode": cfg.ingest_source})
    companies, comp_source = sources.load_companies(cfg)
    actions, act_source = sources.load_corp_actions(cfg)
    source_label = "live" if "live" in {comp_source, act_source} else "sample"

    rows_ok = 0
    rows_failed = 0
    error: str | None = None
    run_id: int | None = None
    try:
        with db.connect(cfg) as conn:
            run_id = _start_run(conn, source_label)
            try:
                n_comp = _upsert_companies(conn, companies, comp_source)
                n_act = _upsert_corp_actions(conn, actions, act_source)
                # F5: also load sample fundamentals when using sample mode
                n_fund = 0
                if source_label == "sample":
                    from projects.finance_agent import fundamentals as fund

                    n_fund = int(fund.ingest_sample(config=cfg).get("rows_ok", 0))
                rows_ok = n_comp + n_act + n_fund
                _finish_run(
                    conn,
                    run_id,
                    status="success",
                    rows_ok=rows_ok,
                    rows_failed=0,
                    file_names=[
                        "sample_company_master.csv",
                        "sample_corp_actions.csv",
                        "sample_fundamentals.csv",
                    ]
                    if source_label == "sample"
                    else ["live"],
                    error=None,
                    metadata={
                        "companies": n_comp,
                        "corp_actions": n_act,
                        "fundamentals": n_fund,
                        "company_source": comp_source,
                        "action_source": act_source,
                    },
                )
                log.info(
                    "ingest_success",
                    extra={
                        "run_id": run_id,
                        "companies": n_comp,
                        "corp_actions": n_act,
                        "fundamentals": n_fund,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                rows_failed = len(companies) + len(actions)
                error = str(exc)
                _finish_run(
                    conn,
                    run_id,
                    status="failed",
                    rows_ok=0,
                    rows_failed=rows_failed,
                    file_names=[],
                    error=error,
                    metadata={},
                )
                log.error("ingest_failed", extra={"run_id": run_id, "error": error})
                raise
    finally:
        if have_lock:
            try:
                redis_client.release_lock(LOCK_NAME, token=token, config=cfg)
            except Exception as exc:  # noqa: BLE001
                log.warning("redis_unlock_failed", extra={"error": str(exc)})

    return {
        "skipped": False,
        "run_id": run_id,
        "source": source_label,
        "rows_ok": rows_ok,
        "rows_failed": rows_failed,
        "error": error,
    }


def status(config: FinanceConfig | None = None) -> dict[str, Any]:
    cfg = config or load_config()
    with db.connect(cfg) as conn:
        companies = conn.execute("SELECT count(*) FROM finance_company_master").fetchone()[0]
        actions = conn.execute("SELECT count(*) FROM finance_corp_actions").fetchone()[0]
        fundamentals = conn.execute("SELECT count(*) FROM finance_fundamentals").fetchone()[0]
        by_type = conn.execute(
            """
            SELECT action_type, count(*)
            FROM finance_corp_actions
            GROUP BY action_type
            ORDER BY action_type
            """
        ).fetchall()
        last = conn.execute(
            """
            SELECT id, source, status, rows_ok, rows_failed, started_at, finished_at
            FROM finance_ingest_runs
            ORDER BY started_at DESC
            LIMIT 1
            """
        ).fetchone()
    return {
        "companies": int(companies),
        "corp_actions": int(actions),
        "fundamentals": int(fundamentals),
        "corp_actions_by_type": {t: int(c) for t, c in by_type},
        "last_run": (
            {
                "id": last[0],
                "source": last[1],
                "status": last[2],
                "rows_ok": last[3],
                "rows_failed": last[4],
                "started_at": str(last[5]),
                "finished_at": str(last[6]) if last[6] else None,
            }
            if last
            else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finance F2 daily NSE/BSE ingest")
    parser.add_argument("--status", action="store_true", help="Show counts + last run")
    args = parser.parse_args(argv)

    if args.status:
        print(json.dumps(status(), indent=2))
        return 0

    result = run_ingest()
    print(json.dumps(result, indent=2, default=str))
    if result.get("skipped"):
        return 0
    if result.get("error"):
        return 1
    print(json.dumps({"status": status()}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
