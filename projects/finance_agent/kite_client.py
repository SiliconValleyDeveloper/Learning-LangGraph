"""Zerodha Kite market-data client (quotes only — no orders).

Modes (FINANCE_QUOTE_SOURCE):
  auto   — use Kite when api_key + access_token are set; else sample
  kite   — require live Kite (fail if missing / library missing)
  sample — always use bundled sample_quotes.csv (offline)

Analysis platform only: never place buy/sell orders.
"""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from projects.finance_agent.config import FinanceConfig, load_config
from projects.finance_agent.logging_util import get_logger

log = get_logger("finance.kite")
DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_QUOTES = DATA_DIR / "sample_quotes.csv"


@dataclass
class Quote:
    symbol: str
    exchange: str
    ltp: float
    volume: int | None = None
    oi: int | None = None
    bid: float | None = None
    ask: float | None = None
    change_pct: float | None = None
    name: str | None = None
    source: str = "sample"
    ts: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def kite_credentials_ready(config: FinanceConfig | None = None) -> bool:
    cfg = config or load_config()
    return bool(cfg.kite_api_key and cfg.kite_access_token)


def resolve_quote_source(config: FinanceConfig | None = None) -> str:
    cfg = config or load_config()
    mode = cfg.quote_source
    if mode == "sample":
        return "sample"
    if mode == "kite":
        return "kite"
    # auto
    return "kite" if kite_credentials_ready(cfg) else "sample"


def _instrument_key(exchange: str, symbol: str) -> str:
    return f"{exchange}:{symbol}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_sample_quotes(
    symbols: list[str] | None = None,
    *,
    jitter: bool = True,
) -> list[Quote]:
    """Load sample quotes; optional tiny jitter so refreshes look live."""
    text = SAMPLE_QUOTES.read_text(encoding="utf-8")
    want = {s.upper() for s in symbols} if symbols else None
    out: list[Quote] = []
    for raw in csv.DictReader(text.splitlines()):
        symbol = (raw.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        if want is not None and symbol not in want:
            continue
        exchange = (raw.get("exchange") or "NSE").strip().upper()
        try:
            ltp = float(raw.get("ltp") or 0)
        except ValueError:
            continue
        if jitter and ltp > 0:
            # ±0.15% so repeated refreshes show movement without looking random chaos
            ltp = round(ltp * (1.0 + random.uniform(-0.0015, 0.0015)), 2)

        def _opt_float(key: str) -> float | None:
            v = (raw.get(key) or "").strip()
            if not v:
                return None
            try:
                return float(v)
            except ValueError:
                return None

        def _opt_int(key: str) -> int | None:
            v = (raw.get(key) or "").strip()
            if not v:
                return None
            try:
                return int(float(v))
            except ValueError:
                return None

        out.append(
            Quote(
                symbol=symbol,
                exchange=exchange,
                ltp=ltp,
                volume=_opt_int("volume"),
                oi=_opt_int("oi"),
                bid=_opt_float("bid"),
                ask=_opt_float("ask"),
                change_pct=_opt_float("change_pct"),
                name=(raw.get("name") or "").strip() or None,
                source="sample",
                ts=_now_iso(),
            )
        )
    return out


def _fetch_kite_quotes(
    symbols: list[str],
    *,
    config: FinanceConfig,
    exchange: str = "NSE",
) -> list[Quote]:
    try:
        from kiteconnect import KiteConnect
    except ImportError as exc:
        raise RuntimeError(
            "kiteconnect not installed. pip install kiteconnect  "
            "or set FINANCE_QUOTE_SOURCE=sample"
        ) from exc

    if not kite_credentials_ready(config):
        raise RuntimeError(
            "KITE_API_KEY and KITE_ACCESS_TOKEN required for live quotes"
        )

    kite = KiteConnect(api_key=config.kite_api_key)
    kite.set_access_token(config.kite_access_token)

    # Allow already-qualified keys like NSE:RELIANCE
    keys: list[str] = []
    for s in symbols:
        s = s.strip().upper()
        if ":" in s:
            keys.append(s)
        else:
            keys.append(_instrument_key(exchange, s))

    raw = kite.quote(keys)
    out: list[Quote] = []
    for key, payload in (raw or {}).items():
        parts = key.split(":", 1)
        ex = parts[0] if len(parts) == 2 else exchange
        sym = parts[1] if len(parts) == 2 else key
        depth = (payload or {}).get("depth") or {}
        buy = (depth.get("buy") or [{}])[0] if depth.get("buy") else {}
        sell = (depth.get("sell") or [{}])[0] if depth.get("sell") else {}
        ohlc = (payload or {}).get("ohlc") or {}
        last = float((payload or {}).get("last_price") or 0)
        prev = float(ohlc.get("close") or 0) or None
        change_pct = None
        if prev and prev > 0 and last:
            change_pct = round(((last - prev) / prev) * 100.0, 4)
        out.append(
            Quote(
                symbol=sym,
                exchange=ex,
                ltp=last,
                volume=int((payload or {}).get("volume") or 0) or None,
                oi=int((payload or {}).get("oi") or 0) or None,
                bid=float(buy.get("price") or 0) or None,
                ask=float(sell.get("price") or 0) or None,
                change_pct=change_pct,
                name=((payload or {}).get("instrument_token") and sym) or sym,
                source="kite",
                ts=_now_iso(),
            )
        )
    log.info("kite_quotes_ok", extra={"count": len(out)})
    return out


def fetch_quotes(
    symbols: list[str] | None = None,
    *,
    config: FinanceConfig | None = None,
    exchange: str = "NSE",
) -> tuple[list[Quote], str]:
    """Fetch quotes from Kite or sample. Returns (quotes, source_used)."""
    cfg = config or load_config()
    syms = [s.upper() for s in (symbols or list(cfg.quote_default_symbols))]
    mode = resolve_quote_source(cfg)

    if mode == "kite":
        try:
            return _fetch_kite_quotes(syms, config=cfg, exchange=exchange), "kite"
        except Exception as exc:  # noqa: BLE001
            if cfg.quote_source == "kite":
                raise
            log.warning("kite_fallback_sample", extra={"error": str(exc)})
            return load_sample_quotes(syms), "sample"

    return load_sample_quotes(syms), "sample"
