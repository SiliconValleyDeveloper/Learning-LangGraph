"""
Phase 12 · Lesson 1 — Enterprise RAG in three layers

Interview frame (use this in system-design answers):
  1. Knowledge layer  — where truth lives (docs, ACLs, indexes)
  2. Retrieval layer  — how we find evidence (dense / sparse / hybrid)
  3. Validation layer — how we stay safe (grade, cite, refuse, eval)

Run:
    python Learning/12_rag_architect/01_enterprise_layers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import load_documents, short_preview


def main() -> None:
    docs = load_documents()
    print("=" * 60)
    print("Enterprise RAG — three-layer design (Contoso Ops KB)")
    print("=" * 60)

    print("\n[1] Knowledge layer")
    print("    Sources indexed:")
    for doc in docs:
        print(f"    - {doc.metadata['source']} ({len(doc.page_content)} chars)")
    print("    ACLs: visibility=internal (confidential filtered at retrieve time)")
    print("    Artifacts: chunks, dense embeddings, BM25 postings, optional entity graph")

    print("\n[2] Retrieval layer")
    print("    Query rewrite / multi-query / HyDE → candidate generation")
    print("    Dense (semantic) + sparse BM25 (keywords/IDs) → RRF fuse → rerank")
    print("    Example hard query: ticket IDs like HR-LEAVE-24, ACC-PROD-WRITE")

    print("\n[3] Validation layer")
    print("    Grade evidence → generate with citations → verify faithfulness")
    print("    Offline eval (context recall / faithfulness) before shipping changes")
    print("    Refuse when evidence is weak — never invent policy from memory")

    print("\nSample knowledge preview:")
    print(" ", short_preview(docs[0].page_content, 140))

    print("\nInterview one-liner:")
    print(
        "  'An enterprise RAG pipeline retrieves internal knowledge, validates it "
        "with citations/confidence, and produces safe context-aware answers.'"
    )


if __name__ == "__main__":
    main()
