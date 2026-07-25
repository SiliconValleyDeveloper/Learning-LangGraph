"""Redis helpers for finance agent (F1+): cache, locks, queues."""

from __future__ import annotations

from typing import Any

from projects.finance_agent.config import FinanceConfig, load_config


def _client(config: FinanceConfig | None = None):
    try:
        import redis
    except ImportError as exc:
        raise RuntimeError("Install redis: pip install redis") from exc

    cfg = config or load_config()
    return redis.Redis.from_url(cfg.redis_url, decode_responses=True)


def ping(config: FinanceConfig | None = None) -> bool:
    return _client(config).ping() is True


def get_json(key: str, *, config: FinanceConfig | None = None) -> str | None:
    value = _client(config).get(key)
    return value if isinstance(value, str) else None


def set_json(
    key: str,
    value: str,
    *,
    ttl_seconds: int | None = None,
    config: FinanceConfig | None = None,
) -> None:
    client = _client(config)
    if ttl_seconds is None:
        client.set(key, value)
    else:
        client.setex(key, ttl_seconds, value)


def acquire_lock(
    name: str,
    *,
    ttl_seconds: int = 3600,
    token: str = "1",
    config: FinanceConfig | None = None,
) -> bool:
    """SET NX EX — returns True if this caller holds the lock."""
    return bool(
        _client(config).set(name, token, nx=True, ex=ttl_seconds)
    )


def release_lock(
    name: str,
    *,
    token: str = "1",
    config: FinanceConfig | None = None,
) -> bool:
    client = _client(config)
    current = client.get(name)
    if current == token:
        client.delete(name)
        return True
    return False


def incr(key: str, *, ttl_seconds: int | None = None, config: FinanceConfig | None = None) -> int:
    """INCR key; set TTL when the counter is first created."""
    client = _client(config)
    count = int(client.incr(key))
    if count == 1 and ttl_seconds is not None:
        client.expire(key, ttl_seconds)
    return count


def publish(channel: str, message: str, *, config: FinanceConfig | None = None) -> int:
    """Publish to a Redis pub/sub channel. Returns subscriber count."""
    return int(_client(config).publish(channel, message))


def health(config: FinanceConfig | None = None) -> dict[str, Any]:
    cfg = config or load_config()
    try:
        ok = ping(cfg)
        return {"ok": ok, "redis_url": cfg.redis_url}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "redis_url": cfg.redis_url, "error": str(exc)}
