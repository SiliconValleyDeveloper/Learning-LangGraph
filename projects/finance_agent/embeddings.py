"""Embedding helpers for finance filings (F6).

Prefer Ollama ``nomic-embed-text`` (768-d, matches ``finance_chunks.embedding``).
Falls back to a deterministic hash embedding when Ollama is unreachable so
sample ingest / lexical demos still work offline.
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Literal

from projects.finance_agent.config import FinanceConfig, load_config
from projects.finance_agent.logging_util import get_logger

log = get_logger("finance.embeddings")

EmbedBackend = Literal["ollama", "hash"]


def _as_vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


def hash_embed(text: str, *, dims: int = 768) -> list[float]:
    """Cheap deterministic bag-of-tokens embedding (L2-normalised)."""
    vec = [0.0] * dims
    tokens = re.findall(r"[a-z0-9]{3,}", (text or "").lower())
    if not tokens:
        tokens = ["empty"]
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@lru_cache(maxsize=1)
def _ollama_client(base_url: str, model: str):
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=model, base_url=base_url)


def _ollama_ok(cfg: FinanceConfig) -> bool:
    try:
        import urllib.request

        url = cfg.ollama_base_url.rstrip("/") + "/api/tags"
        with urllib.request.urlopen(url, timeout=1.5) as resp:  # noqa: S310
            return int(getattr(resp, "status", 200) or 200) < 400
    except Exception:  # noqa: BLE001
        return False


def resolve_backend(config: FinanceConfig | None = None) -> EmbedBackend:
    cfg = config or load_config()
    mode = cfg.embed_backend
    if mode == "hash":
        return "hash"
    if mode == "ollama":
        return "ollama"
    # auto
    return "ollama" if _ollama_ok(cfg) else "hash"


def embed_texts(
    texts: list[str],
    *,
    config: FinanceConfig | None = None,
) -> tuple[list[list[float]], EmbedBackend]:
    cfg = config or load_config()
    backend = resolve_backend(cfg)
    if not texts:
        return [], backend
    if backend == "ollama":
        try:
            client = _ollama_client(cfg.ollama_base_url, cfg.embedding_model)
            vectors = client.embed_documents(texts)
            if vectors and len(vectors[0]) != cfg.embed_dims:
                log.warning(
                    "embed_dims_mismatch",
                    extra={"got": len(vectors[0]), "expected": cfg.embed_dims},
                )
            return [list(map(float, v)) for v in vectors], "ollama"
        except Exception as exc:  # noqa: BLE001
            log.warning("ollama_embed_fallback", extra={"error": str(exc)})
            backend = "hash"
    return [hash_embed(t, dims=cfg.embed_dims) for t in texts], backend


def embed_query(
    text: str,
    *,
    config: FinanceConfig | None = None,
) -> tuple[list[float], EmbedBackend]:
    vectors, backend = embed_texts([text], config=config)
    return vectors[0], backend


def vector_literal(
    text: str,
    *,
    config: FinanceConfig | None = None,
) -> tuple[str, EmbedBackend]:
    vec, backend = embed_query(text, config=config)
    return _as_vector_literal(vec), backend


def vectors_as_literals(vectors: list[list[float]]) -> list[str]:
    return [_as_vector_literal(v) for v in vectors]
