"""
Phase 12 · Lesson 4 — HyDE + query optimization

HyDE: generate a hypothetical answer, embed THAT, then retrieve.
Helps zero-shot / vague questions where the query vocabulary ≠ doc vocabulary.

Also demos multi-query rewrite (ask the LLM for alternate search phrasings).

Needs: Ollama with qwen3:8b and nomic-embed-text

Run:
    python Learning/12_rag_architect/04_hyde_and_query_opt.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from helpers import dense_search, short_preview
from llm import get_llm, require_ollama


def _clean(text: object) -> str:
    raw = text if isinstance(text, str) else str(text)
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def multi_query(question: str) -> list[str]:
    prompt = [
        SystemMessage(
            content=(
                "Rewrite the user question into 3 short search queries for an "
                "enterprise knowledge base. Return one query per line. No numbering."
            )
        ),
        HumanMessage(content=question),
    ]
    text = _clean(get_llm(temperature=0.2).invoke(prompt).content)
    lines = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()]
    return lines[:3] or [question]


def hyde_passage(question: str) -> str:
    prompt = [
        SystemMessage(
            content=(
                "Write a short hypothetical handbook paragraph that would answer "
                "the question. Do not say you are hypothesizing. 3-5 sentences."
            )
        ),
        HumanMessage(content=question),
    ]
    return _clean(get_llm(temperature=0.4).invoke(prompt).content)


def main() -> None:
    require_ollama()
    question = "How fast must we respond when customers cannot use the product?"
    print(f"Question: {question}\n")

    print("Baseline dense retrieval:")
    for i, (doc, score) in enumerate(dense_search(question, k=3), start=1):
        print(f"  {i}. {score:.3f} {doc.metadata.get('source')} | {short_preview(doc.page_content)}")

    print("\nMulti-query rewrites:")
    queries = multi_query(question)
    for q in queries:
        print(f"  - {q}")

    print("\nHyDE hypothetical passage:")
    hypo = hyde_passage(question)
    print(f"  {short_preview(hypo, 220)}")

    print("\nDense retrieval using HyDE passage as the query:")
    for i, (doc, score) in enumerate(dense_search(hypo, k=3), start=1):
        print(f"  {i}. {score:.3f} {doc.metadata.get('source')} | {short_preview(doc.page_content)}")

    print("\nInterview tip:")
    print("  Use HyDE when users ask vaguely; skip it for exact ticket/ID lookups.")


if __name__ == "__main__":
    main()
