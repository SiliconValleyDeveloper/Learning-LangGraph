"""Store factory — memory today, pgvector when enabled."""

from __future__ import annotations

from functools import lru_cache

from projects.advanced_chatbot.config import load_config
from projects.advanced_chatbot.store.base import DocumentVectorStore


@lru_cache(maxsize=1)
def get_store() -> DocumentVectorStore:
    config = load_config()
    if config.vector_backend == "pgvector":
        from projects.advanced_chatbot.store.pgvector import PgVectorDocumentStore

        return PgVectorDocumentStore(config)
    from projects.advanced_chatbot.store.memory import MemoryDocumentStore

    return MemoryDocumentStore(config)


def reset_store() -> None:
    """Clear cached store (useful after env change / tests)."""
    get_store.cache_clear()
