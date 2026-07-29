"""Runtime configuration for the shipping/logistics project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env", override=False)


@dataclass(frozen=True)
class ShippingConfig:
    database_url: str
    schema: str
    currency: str
    quote_valid_days: int
    approval_required: bool
    checkpoint_db: Path
    use_llm_answers: bool
    rag_top_k: int
    max_retrieval_retries: int
    max_fix_attempts: int


def load_config() -> ShippingConfig:
    return ShippingConfig(
        database_url=os.getenv(
            "SHIPPING_DATABASE_URL",
            os.getenv(
                "DATABASE_URL",
                "postgresql://langgraph:langgraph@localhost:5433/langgraph",
            ),
        ),
        schema=os.getenv("SHIPPING_SCHEMA", "shipping"),
        currency=os.getenv("SHIPPING_CURRENCY", "USD"),
        quote_valid_days=int(os.getenv("SHIPPING_QUOTE_VALID_DAYS", "14")),
        approval_required=os.getenv(
            "SHIPPING_APPROVAL_REQUIRED", "true"
        ).strip().lower()
        in {"1", "true", "yes", "on"},
        checkpoint_db=Path(
            os.getenv(
                "SHIPPING_CHECKPOINT_DB",
                str(_ROOT / ".data" / "shipping-checkpoints.db"),
            )
        ),
        use_llm_answers=os.getenv(
            "SHIPPING_USE_LLM_ANSWERS", "true"
        ).strip().lower()
        in {"1", "true", "yes", "on"},
        rag_top_k=max(1, min(int(os.getenv("SHIPPING_RAG_TOP_K", "5")), 10)),
        max_retrieval_retries=max(
            0,
            min(int(os.getenv("SHIPPING_MAX_RETRIEVAL_RETRIES", "1")), 3),
        ),
        max_fix_attempts=max(
            0,
            min(int(os.getenv("SHIPPING_MAX_FIX_ATTEMPTS", "1")), 3),
        ),
    )

