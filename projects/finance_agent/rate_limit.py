"""Redis rate limiting for finance API consumers (F3.5)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from projects.finance_agent import redis_client
from projects.finance_agent.config import FinanceConfig, load_config
from projects.finance_agent.logging_util import get_logger

if TYPE_CHECKING:
    from projects.finance_agent.auth import Consumer

log = get_logger("finance.rate_limit")


def check_rate_limit(
    consumer: Consumer,
    *,
    config: FinanceConfig | None = None,
) -> tuple[bool, int | None]:
    """Return (allowed, retry_after_seconds). Uses a fixed 60s window counter."""
    cfg = config or load_config()
    window = int(time.time() // 60)
    key = f"finance:rl:{consumer.id}:{window}"
    try:
        count = redis_client.incr(key, ttl_seconds=70, config=cfg)
        limit = max(1, int(consumer.rate_limit_rpm))
        if count > limit:
            retry = 60 - int(time.time() % 60)
            log.warning(
                "rate_limited",
                extra={"consumer_id": consumer.id, "count": count, "limit": limit},
            )
            return False, max(1, retry)
        return True, None
    except Exception as exc:  # noqa: BLE001
        log.warning("rate_limit_redis_error", extra={"error": str(exc)})
        return True, None
