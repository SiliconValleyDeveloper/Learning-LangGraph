"""Enterprise KB RAG architect lab — hybrid, HyDE, CRAG, Graph RAG, eval."""

from __future__ import annotations

__all__ = ["ask", "ingest_seed", "evaluate"]


def ask(*args, **kwargs):
    from projects.rag_architect.service import ask as _ask

    return _ask(*args, **kwargs)


def ingest_seed(*args, **kwargs):
    from projects.rag_architect.service import ingest_seed as _ingest

    return _ingest(*args, **kwargs)


def evaluate(*args, **kwargs):
    from projects.rag_architect.evaluate import run_eval

    return run_eval(*args, **kwargs)
