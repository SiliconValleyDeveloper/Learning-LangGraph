"""Dense, BM25, hybrid (RRF), HyDE, and light lexical rerank."""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from Learning.llm import get_llm
from projects.rag_architect.config import load_config
from projects.rag_architect.ingest import get_index, tokenize
from projects.rag_architect.models import ChunkHit


def _clean_llm(text: object) -> str:
    raw = text if isinstance(text, str) else str(text)
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def _to_hits(pairs: list[tuple[Document, float]]) -> list[ChunkHit]:
    hits: list[ChunkHit] = []
    for doc, score in pairs:
        hits.append(
            ChunkHit(
                chunk_id=str(doc.metadata.get("chunk_id", "")),
                source=str(doc.metadata.get("source", "unknown")),
                content=doc.page_content,
                score=float(score),
                metadata=dict(doc.metadata),
            )
        )
    return hits


def dense_search(query: str, *, k: int | None = None) -> list[ChunkHit]:
    cfg = load_config()
    k = k or cfg.top_k
    index = get_index()
    assert index.vector is not None
    pairs = index.vector.similarity_search_with_score(query, k=k)
    return _to_hits([(d, float(s)) for d, s in pairs])


def sparse_search(query: str, *, k: int | None = None) -> list[ChunkHit]:
    cfg = load_config()
    k = k or cfg.top_k
    index = get_index()
    assert index.bm25 is not None
    return _to_hits(index.bm25.search(query, k=k))


def rrf_fuse(lists: list[list[ChunkHit]], *, k: int, rrf_k: int = 60) -> list[ChunkHit]:
    scores: dict[str, float] = {}
    docs: dict[str, ChunkHit] = {}
    for ranked in lists:
        for rank, hit in enumerate(ranked, start=1):
            key = hit.chunk_id or f"{hit.source}:{hash(hit.content)}"
            docs[key] = hit
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused: list[ChunkHit] = []
    for key, score in ordered[:k]:
        hit = docs[key]
        fused.append(
            ChunkHit(
                chunk_id=hit.chunk_id,
                source=hit.source,
                content=hit.content,
                score=score,
                metadata=hit.metadata,
            )
        )
    return fused


def hybrid_search(query: str, *, k: int | None = None) -> list[ChunkHit]:
    cfg = load_config()
    k = k or cfg.top_k
    cand = cfg.retrieve_candidates
    return rrf_fuse(
        [dense_search(query, k=cand), sparse_search(query, k=cand)],
        k=k,
    )


def lexical_rerank(query: str, hits: list[ChunkHit], *, k: int | None = None) -> list[ChunkHit]:
    cfg = load_config()
    k = k or cfg.top_k
    q = set(tokenize(query))
    scored: list[ChunkHit] = []
    for hit in hits:
        c = set(tokenize(hit.content))
        overlap = len(q & c) / (len(q) or 1)
        scored.append(
            ChunkHit(
                chunk_id=hit.chunk_id,
                source=hit.source,
                content=hit.content,
                score=0.55 * hit.score + 0.45 * overlap,
                metadata=hit.metadata,
            )
        )
    scored.sort(key=lambda h: h.score, reverse=True)
    return scored[:k]


def hyde_passage(question: str) -> str:
    prompt = [
        SystemMessage(
            content=(
                "Write a short hypothetical internal handbook paragraph that would "
                "answer the question. Do not say it is hypothetical. 3-5 sentences."
            )
        ),
        HumanMessage(content=question),
    ]
    return _clean_llm(get_llm(temperature=0.4).invoke(prompt).content)


def hyde_search(question: str, *, k: int | None = None) -> tuple[list[ChunkHit], str]:
    passage = hyde_passage(question)
    return dense_search(passage, k=k), passage


def multi_query(question: str) -> list[str]:
    prompt = [
        SystemMessage(
            content=(
                "Rewrite into 3 short enterprise KB search queries. "
                "One query per line. No numbering."
            )
        ),
        HumanMessage(content=question),
    ]
    text = _clean_llm(get_llm(temperature=0.2).invoke(prompt).content)
    lines = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()]
    return lines[:3] or [question]


def format_context(hits: list[ChunkHit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[{i}] Source: {hit.source}\n{hit.content}")
    return "\n\n".join(blocks)
