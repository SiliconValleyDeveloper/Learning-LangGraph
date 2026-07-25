"""
Phase 12 · Lesson 7 — Evaluate RAG (simplified RAGAS-style)

Metrics (local, no RAGAS install required):
  - context_recall: gold keywords appear in retrieved context
  - faithfulness:   answer tokens overlap retrieved context (proxy)
  - answer_has_citation: answer contains [n] style citations

Needs: Ollama with qwen3:8b and nomic-embed-text

Run:
    python Learning/12_rag_architect/07_evaluate_rag.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from helpers import format_context, hybrid_search, tokenize
from llm import get_llm, require_ollama

GOLD_PATH = Path(__file__).resolve().parent / "eval" / "gold_qa.json"


def _clean(text: object) -> str:
    raw = text if isinstance(text, str) else str(text)
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def context_recall(context: str, must_include: list[str]) -> float:
    if not must_include:
        return 1.0
    lowered = context.lower()
    hits = sum(1 for key in must_include if key.lower() in lowered)
    return hits / len(must_include)


def faithfulness_proxy(answer: str, context: str) -> float:
    a = set(tokenize(answer))
    c = set(tokenize(context))
    if not a:
        return 0.0
    return len(a & c) / len(a)


def has_citation(answer: str) -> bool:
    return bool(re.search(r"\[\d+\]", answer))


def answer_question(question: str, context: str) -> str:
    prompt = [
        SystemMessage(
            content=(
                "Answer using ONLY the context. Cite sources like [1]. "
                "If insufficient, say you do not know."
            )
        ),
        HumanMessage(content=f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"),
    ]
    return _clean(get_llm(temperature=0).invoke(prompt).content)


def main() -> None:
    require_ollama()
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    rows = []
    for item in gold:
        docs = hybrid_search(item["question"], k=4)
        context = format_context(docs)
        answer = answer_question(item["question"], context)
        recall = context_recall(context, item.get("must_include", []))
        faith = faithfulness_proxy(answer, context)
        cited = has_citation(answer)
        rows.append((item["id"], recall, faith, cited, answer))

    print(f"{'id':<16} {'recall':>7} {'faith':>7} {'cite':>5}")
    print("-" * 40)
    for rid, recall, faith, cited, _ in rows:
        print(f"{rid:<16} {recall:>7.2f} {faith:>7.2f} {str(cited):>5}")

    avg_recall = sum(r[1] for r in rows) / len(rows)
    avg_faith = sum(r[2] for r in rows) / len(rows)
    cite_rate = sum(1 for r in rows if r[3]) / len(rows)
    print("-" * 40)
    print(f"{'AVG':<16} {avg_recall:>7.2f} {avg_faith:>7.2f} {cite_rate:>5.2f}")
    print("\nInterview tip: measure retrieval and generation separately before tuning.")


if __name__ == "__main__":
    main()
