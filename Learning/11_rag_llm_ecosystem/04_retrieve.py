"""
Phase 11 · Lesson 4 — Semantic retrieval

What you will learn
-------------------
1. A vector store keeps chunk vectors and metadata
2. The question is embedded in the same vector space
3. Similarity search returns the top-k chunks
4. Retrieval finds context; it does not generate the final answer

Run:
    python Learning/11_rag_llm_ecosystem/04_retrieve.py
"""

from __future__ import annotations

from rag_helpers import build_vector_store


if __name__ == "__main__":
    question = "How does a conditional edge differ from a normal edge?"
    store = build_vector_store()

    print(f"Question: {question}\n")
    results = store.similarity_search_with_score(question, k=3)

    for rank, (document, score) in enumerate(results, start=1):
        source = document.metadata.get("source", "unknown")
        print(f"#{rank} source={source} score={score:.4f}")
        print(document.page_content[:350])
        print()

    print(
        "The retriever returned evidence, not an answer. "
        "Lesson 5 passes this evidence to the chat model."
    )
