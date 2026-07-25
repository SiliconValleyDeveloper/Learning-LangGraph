"""Lightweight in-memory entity graph for multi-hop enterprise questions."""

from __future__ import annotations

import re
from collections import defaultdict

from projects.rag_architect.ingest import get_index
from projects.rag_architect.models import ChunkHit
from projects.rag_architect.retrieve import dense_search, hybrid_search, rrf_fuse

SEED_ENTITIES = {
    "P1": {"type": "severity", "source": "incident_playbook.md"},
    "P2": {"type": "severity", "source": "incident_playbook.md"},
    "PagerDuty": {"type": "tool", "source": "incident_playbook.md"},
    "contoso-ops-p1": {"type": "service", "source": "incident_playbook.md"},
    "ACC-PROD-WRITE": {"type": "ticket", "source": "access_policy.md"},
    "prod-break-glass": {"type": "role", "source": "access_policy.md"},
    "Okta": {"type": "system", "source": "access_policy.md"},
    "HR-LEAVE-24": {"type": "ticket", "source": "employee_handbook.md"},
    "SEC-101": {"type": "training", "source": "onboarding_faq.md"},
    "ACC-STG": {"type": "ticket", "source": "onboarding_faq.md"},
}

SEED_EDGES = [
    ("P1", "pages", "PagerDuty"),
    ("PagerDuty", "service", "contoso-ops-p1"),
    ("P1", "may_use", "prod-break-glass"),
    ("ACC-PROD-WRITE", "approves", "prod-break-glass"),
    ("Okta", "gates", "ACC-PROD-WRITE"),
    ("HR-LEAVE-24", "used_for", "leave"),
    ("SEC-101", "required_by", "onboarding"),
]


def build_adjacency() -> dict[str, list[tuple[str, str]]]:
    adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for src, rel, dst in SEED_EDGES:
        adj[src].append((rel, dst))
        adj[dst].append((f"rev:{rel}", src))
    return adj


def extract_seeds(question: str) -> list[str]:
    hits: list[str] = []
    lowered = question.lower()
    for name in SEED_ENTITIES:
        if name.lower() in lowered:
            hits.append(name)
    if re.search(r"\bp1\b", lowered) and "P1" not in hits:
        hits.append("P1")
    if "break-glass" in lowered or "break glass" in lowered:
        if "prod-break-glass" not in hits:
            hits.append("prod-break-glass")
    if "leave" in lowered and "HR-LEAVE-24" not in hits:
        hits.append("HR-LEAVE-24")
    if "onboarding" in lowered and "SEC-101" not in hits:
        hits.append("SEC-101")
    return hits


def hop_expand(
    seeds: list[str],
    *,
    hops: int = 1,
) -> tuple[list[str], list[tuple[str, str, str]]]:
    adj = build_adjacency()
    seen = set(seeds)
    frontier = list(seeds)
    edges: list[tuple[str, str, str]] = []
    for _ in range(hops):
        nxt: list[str] = []
        for node in frontier:
            for rel, neighbor in adj.get(node, []):
                edges.append((node, rel, neighbor))
                if neighbor not in seen:
                    seen.add(neighbor)
                    nxt.append(neighbor)
        frontier = nxt
    return sorted(seen), edges


def _chunks_for_sources(sources: set[str], *, query: str) -> list[ChunkHit]:
    index = get_index()
    hits: list[ChunkHit] = []
    for chunk in index.chunks:
        src = str(chunk.metadata.get("source", ""))
        if src not in sources:
            continue
        # prefer chunks that mention any query token or entity-ish tokens
        score = 0.5
        text_l = chunk.page_content.lower()
        for token in re.findall(r"[a-z0-9\-]{3,}", query.lower()):
            if token in text_l:
                score += 0.1
        hits.append(
            ChunkHit(
                chunk_id=str(chunk.metadata.get("chunk_id", "")),
                source=src,
                content=chunk.page_content,
                score=score,
                metadata=dict(chunk.metadata),
            )
        )
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:8]


def graph_search(question: str, *, k: int = 4) -> tuple[list[ChunkHit], list[str]]:
    """Vector/hybrid recall fused with graph-hop source expansion."""
    seeds = extract_seeds(question)
    entities, edges = hop_expand(seeds, hops=1)
    sources = {
        SEED_ENTITIES[e]["source"]
        for e in entities
        if e in SEED_ENTITIES
    }
    notes = [
        f"graph_seeds={seeds}",
        f"graph_entities={entities}",
        f"graph_edges={len(edges)}",
    ]
    vector_hits = hybrid_search(question, k=k)
    if not sources:
        # fall back: still try dense for semantic entry points
        return dense_search(question, k=k), notes + ["graph_fallback=dense"]

    graph_hits = _chunks_for_sources(sources, query=question)
    fused = rrf_fuse([vector_hits, graph_hits], k=k)
    return fused, notes
