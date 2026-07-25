"""Data sources for the daily ingest (F2).

Default is bundled **sample** CSVs (offline, no licensing concern).
Set FINANCE_INGEST_SOURCE=live + the *_URL env vars to fetch real files;
on any failure the loader degrades gracefully back to the sample data.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from projects.finance_agent.config import FinanceConfig, load_config
from projects.finance_agent.logging_util import get_logger

log = get_logger("finance.sources")
DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_COMPANY = DATA_DIR / "sample_company_master.csv"
SAMPLE_CORP_ACTIONS = DATA_DIR / "sample_corp_actions.csv"

VALID_EXCHANGES = {"NSE", "BSE", "OTHER"}


@dataclass
class CompanyRow:
    symbol: str
    exchange: str
    isin: str | None
    name: str
    series: str | None
    status: str
    sector: str | None
    industry: str | None


@dataclass
class CorpActionRow:
    symbol: str
    exchange: str
    action_type: str
    ex_date: date | None
    record_date: date | None
    ratio: str | None
    amount: float | None


def _parse_date(value: str | None) -> date | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        log.warning("bad_date", extra={"value": value})
        return None


def _norm_exchange(value: str | None) -> str:
    ex = (value or "NSE").strip().upper()
    return ex if ex in VALID_EXCHANGES else "OTHER"


def _read_csv_text(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def _maybe_download(url: str | None) -> str | None:
    if not url:
        return None
    try:
        import httpx

        headers = {"User-Agent": "finance-agent/0.1 (analysis)"}
        with httpx.Client(timeout=30.0, headers=headers, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            log.info("download_ok", extra={"url": url, "bytes": len(resp.content)})
            return resp.text
    except Exception as exc:  # noqa: BLE001
        log.warning("download_failed", extra={"url": url, "error": str(exc)})
        return None


def load_companies(config: FinanceConfig | None = None) -> tuple[list[CompanyRow], str]:
    cfg = config or load_config()
    text = None
    source = "sample"
    if cfg.ingest_source == "live":
        text = _maybe_download(cfg.nse_equity_url) or _maybe_download(cfg.bse_equity_url)
        if text is not None:
            source = "live"
    if text is None:
        text = SAMPLE_COMPANY.read_text(encoding="utf-8")

    rows: list[CompanyRow] = []
    for raw in _read_csv_text(text):
        symbol = (raw.get("symbol") or "").strip().upper()
        name = (raw.get("name") or "").strip()
        if not symbol or not name:
            continue
        rows.append(
            CompanyRow(
                symbol=symbol,
                exchange=_norm_exchange(raw.get("exchange")),
                isin=(raw.get("isin") or "").strip() or None,
                name=name,
                series=(raw.get("series") or "").strip() or None,
                status=(raw.get("status") or "active").strip().lower() or "active",
                sector=(raw.get("sector") or "").strip() or None,
                industry=(raw.get("industry") or "").strip() or None,
            )
        )
    log.info("companies_loaded", extra={"count": len(rows), "source": source})
    return rows, source


def load_corp_actions(
    config: FinanceConfig | None = None,
) -> tuple[list[CorpActionRow], str]:
    cfg = config or load_config()
    text = None
    source = "sample"
    if cfg.ingest_source == "live":
        text = _maybe_download(cfg.corp_actions_url)
        if text is not None:
            source = "live"
    if text is None:
        text = SAMPLE_CORP_ACTIONS.read_text(encoding="utf-8")

    rows: list[CorpActionRow] = []
    for raw in _read_csv_text(text):
        symbol = (raw.get("symbol") or "").strip().upper()
        action_type = (raw.get("action_type") or "").strip().lower()
        if not symbol or not action_type:
            continue
        amount_raw = (raw.get("amount") or "").strip()
        try:
            amount = float(amount_raw) if amount_raw else None
        except ValueError:
            amount = None
        rows.append(
            CorpActionRow(
                symbol=symbol,
                exchange=_norm_exchange(raw.get("exchange")),
                action_type=action_type,
                ex_date=_parse_date(raw.get("ex_date")),
                record_date=_parse_date(raw.get("record_date")),
                ratio=(raw.get("ratio") or "").strip() or None,
                amount=amount,
            )
        )
    log.info("corp_actions_loaded", extra={"count": len(rows), "source": source})
    return rows, source
