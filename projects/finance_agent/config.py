"""Runtime config for finance agent (F1+ will expand).

Reads the same DATABASE_URL / Redis / Kite env vars documented in WORKFLOW.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env", override=False)


@dataclass(frozen=True)
class FinanceConfig:
    enabled: bool
    database_url: str | None
    redis_url: str
    vector_backend: str
    rerank_backend: str
    retrieve_candidates: int
    rerank_top_k: int
    kite_api_key: str | None
    kite_api_secret: str | None
    kite_access_token: str | None
    nse_bse_ingest_enabled: bool
    ingest_source: str          # sample | live
    ingest_lock_ttl: int
    ingest_require_lock: bool
    nse_equity_url: str | None
    bse_equity_url: str | None
    corp_actions_url: str | None
    quote_source: str           # auto | sample | kite
    quote_ttl_seconds: int
    quote_persist_ticks: bool
    quote_default_symbols: tuple[str, ...]
    quote_stream_channel: str
    api_auth_required: bool
    api_key_pepper: str
    api_default_tier: str
    api_default_rpm: int
    ollama_base_url: str
    embedding_model: str
    embed_dims: int
    embed_backend: str  # auto | ollama | hash


def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _symbols(raw: str | None) -> tuple[str, ...]:
    if not raw or not raw.strip():
        return (
            "RELIANCE",
            "TCS",
            "INFY",
            "HDFCBANK",
            "ICICIBANK",
            "SBIN",
            "ITC",
            "LT",
            "HINDUNILVR",
            "BHARTIARTL",
        )
    return tuple(s.strip().upper() for s in raw.split(",") if s.strip())


def load_config() -> FinanceConfig:
    return FinanceConfig(
        enabled=_bool("FINANCE_ENABLED", "false"),
        database_url=os.getenv("DATABASE_URL") or None,
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        vector_backend=os.getenv("VECTOR_BACKEND", "pgvector").strip().lower(),
        rerank_backend=os.getenv("RERANK_BACKEND", "auto").strip().lower(),
        retrieve_candidates=int(os.getenv("RETRIEVE_CANDIDATES", "12")),
        rerank_top_k=int(os.getenv("RERANK_TOP_K", "5")),
        kite_api_key=os.getenv("KITE_API_KEY") or None,
        kite_api_secret=os.getenv("KITE_API_SECRET") or None,
        kite_access_token=os.getenv("KITE_ACCESS_TOKEN") or None,
        nse_bse_ingest_enabled=_bool("NSE_BSE_INGEST_ENABLED", "true"),
        ingest_source=os.getenv("FINANCE_INGEST_SOURCE", "sample").strip().lower(),
        ingest_lock_ttl=int(os.getenv("FINANCE_INGEST_LOCK_TTL", "3600")),
        ingest_require_lock=_bool("FINANCE_INGEST_REQUIRE_LOCK", "false"),
        nse_equity_url=os.getenv("NSE_EQUITY_URL") or None,
        bse_equity_url=os.getenv("BSE_EQUITY_URL") or None,
        corp_actions_url=os.getenv("CORP_ACTIONS_URL") or None,
        quote_source=os.getenv("FINANCE_QUOTE_SOURCE", "auto").strip().lower(),
        quote_ttl_seconds=int(os.getenv("FINANCE_QUOTE_TTL", "15")),
        quote_persist_ticks=_bool("FINANCE_QUOTE_PERSIST_TICKS", "true"),
        quote_default_symbols=_symbols(os.getenv("FINANCE_QUOTE_SYMBOLS")),
        quote_stream_channel=os.getenv(
            "FINANCE_QUOTE_STREAM", "finance:quote:stream"
        ),
        # F3.5: require API key on quote endpoints (status stays public)
        api_auth_required=_bool("FINANCE_API_AUTH_REQUIRED", "true"),
        api_key_pepper=os.getenv("FINANCE_API_KEY_PEPPER", "finance-lab-pepper"),
        api_default_tier=os.getenv("FINANCE_API_DEFAULT_TIER", "free").strip().lower(),
        api_default_rpm=int(os.getenv("FINANCE_API_DEFAULT_RPM", "60")),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        embedding_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        embed_dims=int(os.getenv("EMBED_DIMS", "768")),
        embed_backend=os.getenv("FINANCE_EMBED_BACKEND", "auto").strip().lower(),
    )
