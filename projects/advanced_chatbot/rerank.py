"""Rerank retrieved chunks: broad retrieve → careful re-score → keep top-k.

Default backend is a local lexical + vector blend (no extra install).
If ``sentence-transformers`` is installed and RERANK_BACKEND=auto|cross_encoder,
uses a CrossEncoder for stronger query↔passage scoring.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

from projects.advanced_chatbot.web_search import evidence_overlap

_STOP = {
    "the",
    "and",
    "for",
    "what",
    "how",
    "who",
    "when",
    "where",
    "with",
    "from",
    "that",
    "this",
    "are",
    "was",
    "were",
    "have",
    "has",
    "been",
    "will",
    "can",
    "could",
    "would",
    "should",
    "about",
    "into",
    "your",
    "their",
}


def retrieve_candidate_count() -> int:
    return max(4, int(os.getenv("RETRIEVE_CANDIDATES", "12")))


def rerank_top_k() -> int:
    return max(1, int(os.getenv("RERANK_TOP_K", "5")))


def rerank_backend() -> str:
    """lexical | cross_encoder | auto"""
    return os.getenv("RERANK_BACKEND", "auto").strip().lower()


def _tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
        if t not in _STOP
    }


def _normalize_vector_score(raw: float) -> float:
    """Map store scores into [0, 1] (cosine similarity or distance)."""
    if raw <= 1.5:
        return min(max(float(raw), 0.0), 1.0)
    # Likely a distance: smaller is better.
    return min(max(1.0 - float(raw), 0.0), 1.0)


def _lexical_features(query: str, content: str) -> dict[str, float]:
    overlap = evidence_overlap(query, content)
    q = _tokens(query)
    c = _tokens(content)
    if not q:
        jaccard = 0.0
        coverage = 0.0
    else:
        inter = len(q & c)
        union = len(q | c) or 1
        jaccard = inter / union
        coverage = inter / len(q)
    # Soft boost when query phrases appear as contiguous substrings.
    q_lower = (query or "").lower().strip()
    phrase_hit = 1.0 if q_lower and len(q_lower) > 8 and q_lower in (content or "").lower() else 0.0
    return {
        "overlap": overlap,
        "jaccard": jaccard,
        "coverage": coverage,
        "phrase": phrase_hit,
    }


def lexical_rerank_score(query: str, content: str, vector_score: float) -> float:
    """Fast local rerank score in ~[0, 1]."""
    feats = _lexical_features(query, content)
    vec = _normalize_vector_score(vector_score)
    return round(
        0.35 * feats["overlap"]
        + 0.20 * feats["jaccard"]
        + 0.20 * feats["coverage"]
        + 0.10 * feats["phrase"]
        + 0.15 * vec,
        6,
    )


@lru_cache(maxsize=1)
def _load_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return None
    model_name = os.getenv(
        "RERANK_CROSS_ENCODER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    try:
        return CrossEncoder(model_name)
    except Exception:  # noqa: BLE001 — fall back to lexical
        return None


def _cross_encoder_scores(query: str, passages: list[str]) -> list[float] | None:
    model = _load_cross_encoder()
    if model is None or not passages:
        return None
    pairs = [(query, p) for p in passages]
    raw = model.predict(pairs)
    # Softmax-ish squash to [0, 1] via sigmoid for readability.
    out: list[float] = []
    for value in raw:
        x = float(value)
        # ms-marco scores are unbounded; map with sigmoid.
        out.append(1.0 / (1.0 + pow(2.718281828, -x)))
    return out


def choose_backend() -> str:
    mode = rerank_backend()
    if mode == "lexical":
        return "lexical"
    if mode == "cross_encoder":
        return "cross_encoder" if _load_cross_encoder() is not None else "lexical"
    # auto
    return "cross_encoder" if _load_cross_encoder() is not None else "lexical"


def rerank_chunks(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    top_k: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Return (ranked_top_k_candidates, backend_used).

    Each candidate dict needs at least: source, content, score (vector).
    Output adds: rerank_score, vector_score, rank.
    """
    keep = top_k if top_k is not None else rerank_top_k()
    if not candidates:
        return [], choose_backend()

    backend = choose_backend()
    scored: list[dict[str, Any]] = []

    if backend == "cross_encoder":
        passages = [str(c.get("content") or "") for c in candidates]
        ce_scores = _cross_encoder_scores(query, passages)
        if ce_scores is None:
            backend = "lexical"
        else:
            for cand, ce in zip(candidates, ce_scores):
                vec = float(cand.get("score") or 0.0)
                # Blend a little vector score so embedding signal isn't discarded.
                final = round(0.85 * float(ce) + 0.15 * _normalize_vector_score(vec), 6)
                item = dict(cand)
                item["vector_score"] = round(vec, 4)
                item["rerank_score"] = final
                scored.append(item)

    if backend == "lexical":
        scored = []
        for cand in candidates:
            vec = float(cand.get("score") or 0.0)
            content = str(cand.get("content") or "")
            item = dict(cand)
            item["vector_score"] = round(vec, 4)
            item["rerank_score"] = lexical_rerank_score(query, content, vec)
            scored.append(item)

    scored.sort(key=lambda x: float(x.get("rerank_score") or 0.0), reverse=True)
    top = scored[:keep]
    for index, item in enumerate(top, start=1):
        item["rank"] = index
    return top, backend


def build_context_from_ranked(
    question: str,
    ranked: list[dict[str, Any]],
) -> tuple[str, list[str], list[dict[str, Any]], float]:
    """Build context string, sources, previews, and doc_score from ranked chunks."""
    blocks: list[str] = []
    sources: list[str] = []
    previews: list[dict[str, Any]] = []

    for item in ranked:
        source = str(item.get("source") or "unknown")
        content = str(item.get("content") or "")
        sources.append(source)
        rank = int(item.get("rank") or len(blocks) + 1)
        blocks.append(f"[{rank}] Source: {source}\n{content}")
        previews.append(
            {
                "source": source,
                "score": round(float(item.get("rerank_score") or 0.0), 4),
                "vector_score": round(float(item.get("vector_score") or item.get("score") or 0.0), 4),
                "preview": content[:240],
                "rank": rank,
            }
        )

    context = "\n\n".join(blocks)
    doc_score = doc_score_for_question(question, ranked)
    return context, sorted(set(sources)), previews, doc_score


def doc_score_for_question(question: str, ranked: list[dict[str, Any]]) -> float:
    """Blend best lexical overlap with best rerank score."""
    if not ranked:
        return 0.0
    best_overlap = max(
        evidence_overlap(question, str(item.get("content") or "")) for item in ranked
    )
    best_rerank = max(float(item.get("rerank_score") or 0.0) for item in ranked)
    return round(0.55 * best_overlap + 0.45 * best_rerank, 4)
