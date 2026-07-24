"""
Phase 11 · Lesson 3 — Load, chunk, and embed documents

What you will learn
-------------------
1. Documents contain text plus metadata
2. Chunking trades broad context for retrieval precision
3. Embeddings turn meaning into vectors
4. Chat models and embedding models perform different jobs

Run:
    python Learning/11_rag_llm_ecosystem/03_chunk_embed.py
"""

from __future__ import annotations

import math

from rag_helpers import get_embeddings, load_documents, split_documents


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity without adding a numerical dependency."""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm)


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)

    print(f"Loaded {len(documents)} documents and created {len(chunks)} chunks.\n")
    for index, chunk in enumerate(chunks[:4], start=1):
        print(
            f"[{index}] source={chunk.metadata['source']} "
            f"start={chunk.metadata.get('start_index')} chars={len(chunk.page_content)}"
        )
        print(f"    {chunk.page_content[:110].replace(chr(10), ' ')}...\n")

    embeddings = get_embeddings()
    examples = [
        "LangGraph nodes and edges",
        "A graph has functions connected by control flow",
        "How to bake sourdough bread",
    ]
    vectors = embeddings.embed_documents(examples)

    print(f"Embedding dimensions: {len(vectors[0])}")
    print(
        "Related-text similarity:",
        round(cosine_similarity(vectors[0], vectors[1]), 4),
    )
    print(
        "Unrelated-text similarity:",
        round(cosine_similarity(vectors[0], vectors[2]), 4),
    )
    print("\nHigher cosine similarity usually means closer semantic meaning.")
