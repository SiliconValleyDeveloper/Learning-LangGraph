"""
Phase 12 · Lesson 8 — Vector index tradeoffs (HNSW vs IVF)

No FAISS required — this lesson teaches the interview answers and runs a
tiny in-memory clustering demo that mirrors IVF's "search a few lists" idea.

Run:
    python Learning/12_rag_architect/08_index_tradeoffs.py
"""

from __future__ import annotations

import math
import random
from collections import defaultdict


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a)) or 1.0


def cosine(a: list[float], b: list[float]) -> float:
    return _dot(a, b) / (_norm(a) * _norm(b))


def make_vectors(n: int = 60, dim: int = 8, seed: int = 7) -> list[list[float]]:
    rng = random.Random(seed)
    # three latent clusters
    centers = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
    ]
    vectors = []
    for i in range(n):
        c = centers[i % 3]
        vectors.append([c[j] + rng.uniform(-0.15, 0.15) for j in range(dim)])
    return vectors


def ivf_search(
    vectors: list[list[float]],
    query: list[float],
    *,
    nlist: int = 3,
    nprobe: int = 1,
    k: int = 5,
) -> tuple[list[int], int]:
    """Toy IVF: assign vectors to nearest centroid, search nprobe lists."""
    # centroids = first nlist vectors as a cheap stand-in
    centroids = vectors[:nlist]
    lists: dict[int, list[int]] = defaultdict(list)
    for i, vec in enumerate(vectors):
        best = max(range(nlist), key=lambda c: cosine(vec, centroids[c]))
        lists[best].append(i)

    probe = sorted(range(nlist), key=lambda c: cosine(query, centroids[c]), reverse=True)[
        :nprobe
    ]
    candidates: list[int] = []
    for c in probe:
        candidates.extend(lists[c])
    ranked = sorted(candidates, key=lambda i: cosine(query, vectors[i]), reverse=True)
    return ranked[:k], len(candidates)


def brute_force(vectors: list[list[float]], query: list[float], *, k: int = 5) -> list[int]:
    return sorted(range(len(vectors)), key=lambda i: cosine(query, vectors[i]), reverse=True)[
        :k
    ]


def main() -> None:
    print("=" * 60)
    print("HNSW vs IVF — interview cheat sheet")
    print("=" * 60)
    print(
        """
HNSW (Hierarchical Navigable Small World)
  - Graph of neighbors across layers; greedy hop toward query
  - Best recall/latency for most RAG corpora (< ~50M vectors)
  - Memory heavy; updates can be awkward (tombstones / rebuilds)

IVF (Inverted File)
  - Cluster vectors into lists; search only nprobe lists
  - Cheaper RAM; tunable via nlist / nprobe
  - Often paired with PQ compression at huge scale

Rule of thumb
  - Prototype / moderate prod: HNSW (or pgvector HNSW)
  - Huge / cost-sensitive: IVF-PQ / DiskANN family
  - Always measure recall@k + p95 latency on YOUR data
"""
    )

    vectors = make_vectors()
    query = vectors[0]
    exact = brute_force(vectors, query, k=5)
    ivf_hits, scanned = ivf_search(vectors, query, nlist=3, nprobe=1, k=5)
    overlap = len(set(exact) & set(ivf_hits)) / len(exact)

    print("Tiny IVF demo (not production FAISS):")
    print(f"  corpus={len(vectors)}  exact_top5={exact}")
    print(f"  ivf_top5={ivf_hits}  candidates_scanned={scanned}/{len(vectors)}")
    print(f"  overlap_with_exact={overlap:.2f}")
    print("\nRaise nprobe → better recall, more compute (classic IVF knob).")
    print("Long-context models do NOT make RAG obsolete: cost, recency, ACLs, citations.")


if __name__ == "__main__":
    main()
