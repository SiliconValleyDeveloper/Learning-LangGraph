"""
Phase 12 · Lesson 3 — Dense vs BM25 vs Hybrid (RRF)

Dense embeddings catch paraphrases.
BM25 catches exact tokens (ticket IDs, codes, rare names).
Hybrid (RRF) usually wins in enterprise KBs.

Needs: Ollama with nomic-embed-text

Run:
    python Learning/12_rag_architect/03_hybrid_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from helpers import dense_search, hybrid_search, short_preview, sparse_search
from llm import require_ollama


def _print_hits(title: str, docs: list) -> None:
    print(f"\n{title}")
    for i, doc in enumerate(docs, start=1):
        if isinstance(doc, tuple):
            document, score = doc
            print(
                f"  {i}. score={score:.3f} src={document.metadata.get('source')} "
                f"| {short_preview(document.page_content)}"
            )
        else:
            print(
                f"  {i}. src={doc.metadata.get('source')} "
                f"| {short_preview(doc.page_content)}"
            )


def main() -> None:
    require_ollama()
    # Keyword/ID-heavy query — hybrid should surface the leave ticket code.
    query = "What is the ticket code for leave requests HR-LEAVE?"
    print(f"Query: {query}")

    dense = dense_search(query, k=4)
    sparse = sparse_search(query, k=4)
    hybrid = hybrid_search(query, k=4)

    _print_hits("Dense (semantic)", dense)
    _print_hits("Sparse BM25 (keywords)", sparse)
    _print_hits("Hybrid RRF", hybrid)

    print("\nInterview takeaway:")
    print("  Sparse vs dense vs hybrid — hybrid for enterprise IDs + meaning.")
    print("  Fusion: Reciprocal Rank Fusion is simple and strong without tuning.")


if __name__ == "__main__":
    main()
