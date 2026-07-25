"""
Phase 12 · Lesson 6 — Lightweight Graph RAG

Vector RAG: similarity over chunks
Graph RAG: entities + relations, then hop expansion for multi-hop questions

This lesson builds a tiny in-memory entity graph from the Contoso corpus
(no Neo4j). Good enough to explain the interview difference.

Needs: Ollama with nomic-embed-text

Run:
    python Learning/12_rag_architect/06_graph_rag_light.py
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from helpers import dense_search, load_documents, short_preview
from llm import require_ollama

# Curated entities / edges for a teaching graph (deterministic).
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
}

SEED_EDGES = [
    ("P1", "pages", "PagerDuty"),
    ("PagerDuty", "service", "contoso-ops-p1"),
    ("P1", "requires", "prod-break-glass"),
    ("ACC-PROD-WRITE", "approves", "prod-break-glass"),
    ("Okta", "authenticates", "ACC-PROD-WRITE"),
    ("HR-LEAVE-24", "used_for", "leave"),
]


def build_adjacency() -> dict[str, list[tuple[str, str]]]:
    adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for src, rel, dst in SEED_EDGES:
        adj[src].append((rel, dst))
        adj[dst].append((f"rev:{rel}", src))
    return adj


def extract_seed_hits(question: str) -> list[str]:
    hits = []
    lowered = question.lower()
    for name in SEED_ENTITIES:
        if name.lower() in lowered:
            hits.append(name)
    # soft aliases
    if "p1" in lowered and "P1" not in hits:
        hits.append("P1")
    if "break-glass" in lowered or "break glass" in lowered:
        hits.append("prod-break-glass")
    if "leave" in lowered and "HR-LEAVE-24" not in hits:
        hits.append("HR-LEAVE-24")
    return hits


def hop_expand(seeds: list[str], adj: dict[str, list[tuple[str, str]]], *, hops: int = 1):
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


def chunks_for_entities(entities: list[str]) -> list[str]:
    docs = {d.metadata["source"]: d.page_content for d in load_documents()}
    blocks = []
    for name in entities:
        meta = SEED_ENTITIES.get(name)
        if not meta:
            continue
        source = meta["source"]
        text = docs.get(source, "")
        # keep paragraphs mentioning the entity
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if name.lower() in p.lower()]
        snippet = paras[0] if paras else short_preview(text, 160)
        blocks.append(f"{name} ← {source}\n{snippet}")
    return blocks


def main() -> None:
    require_ollama()
    question = "For a P1, what service do we page and what role is used for break-glass?"
    print(f"Question: {question}\n")

    print("Vector-only top hits:")
    for i, (doc, score) in enumerate(dense_search(question, k=3), start=1):
        print(f"  {i}. {score:.3f} {doc.metadata.get('source')} | {short_preview(doc.page_content)}")

    seeds = extract_seed_hits(question)
    adj = build_adjacency()
    entities, edges = hop_expand(seeds, adj, hops=1)
    print(f"\nGraph seeds: {seeds}")
    print(f"Expanded entities: {entities}")
    print("Edges:")
    for src, rel, dst in edges[:8]:
        print(f"  {src} -[{rel}]-> {dst}")

    print("\nGraph-grounded context snippets:")
    for block in chunks_for_entities(entities):
        print(f"  · {short_preview(block, 160)}")

    print("\nInterview takeaway:")
    print("  Use Graph RAG for multi-hop / relational questions;")
    print("  keep vector RAG for fuzzy semantic lookup; hybridize both in production.")


if __name__ == "__main__":
    main()
